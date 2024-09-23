import sqlite3
import shutil
import os
import pandas as pd
import re

def prepare_res_directory():
    res_dir = './res'
    if os.path.exists(res_dir):
        # Clean the directory by deleting all files
        shutil.rmtree(res_dir)
    # Create the directory
    os.makedirs(res_dir)
    
# Function to process each SQLite file and save results to separate tables in the main SQLite databases
def process_sqlite_file(original_db, comm_conn, compute_conn):
    # Generate processing database filename
    processing_db = 'processing-db.sqlite'

    # Backup the database (copy the original database file to a new file)
    shutil.copyfile(original_db, processing_db)
    print(f"Database {original_db} has been backed up as {processing_db}")

    # Connect to the processing SQLite database
    conn = sqlite3.connect(processing_db)
    cursor = conn.cursor()

    try:
        # Add new columns if they don't exist
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN cudaAPIName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kShortName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kDemangledName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN duration INTEGER;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_time INTEGER DEFAULT 0;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_ratio REAL DEFAULT 0;")

        # Update necessary columns
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET cudaAPIName = (
            SELECT value FROM StringIds
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
            ON r.nameId = StringIds.id
            AND CUPTI_ACTIVITY_KIND_KERNEL.correlationId = r.correlationId);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kShortName = (
            SELECT value FROM StringIds WHERE shortName = StringIds.id);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kDemangledName = (
            SELECT value FROM StringIds WHERE demangledName = StringIds.id);
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET duration = end - start;
        ''')

        cursor.execute('''
        WITH Overlaps AS (
            SELECT 
                c.deviceId AS device_id,
                c.start AS compute_start,
                c.end AS compute_end,
                SUM(
                    CASE
                        WHEN c.start < m.end AND c.end > m.start THEN
                            MIN(c.end, m.end) - MAX(c.start, m.start)
                        ELSE 0
                    END
                ) AS total_overlap_time
            FROM CUPTI_ACTIVITY_KIND_KERNEL c
            JOIN CUPTI_ACTIVITY_KIND_KERNEL m
            ON c.deviceId = m.deviceId
            AND m.kShortName LIKE '%nccl%'
            AND c.kShortName NOT LIKE '%nccl%'
            GROUP BY c.deviceId, c.start, c.end
        )
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL
        SET overlap_time = (
            SELECT total_overlap_time 
            FROM Overlaps 
            WHERE Overlaps.device_id = CUPTI_ACTIVITY_KIND_KERNEL.deviceId
            AND Overlaps.compute_start = CUPTI_ACTIVITY_KIND_KERNEL.start
            AND Overlaps.compute_end = CUPTI_ACTIVITY_KIND_KERNEL.end
        );
        ''')

        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL
        SET overlap_ratio = CAST(overlap_time AS REAL) / duration
        WHERE duration > 0;
        ''')

        conn.commit()

        # Extract the base name of the original database file without extension
        base_name = os.path.splitext(os.path.basename(original_db))[0]

        query_other = '''
        SELECT kShortName, kDemangledName, deviceId, duration, overlap_time, overlap_ratio
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE kShortName IS NOT NULL 
            AND kShortName NOT LIKE '%nccl%'
            AND kDemangledName IS NOT NULL
            AND duration > 0
        '''

        df_other = pd.read_sql_query(query_other, conn)
        grouped = df_other.groupby('deviceId')
        
        for device_id, group in grouped:
            group['id'] = range(1, len(group) + 1)
            cols = ['id'] + [col for col in group.columns if col != 'id']
            group = group[cols]

            table_name = f"{base_name}_device_{device_id}"
            group.to_sql(table_name, compute_conn, if_exists='replace', index=False)

            print(f"Data for device {device_id} has been saved to table {table_name} in the compute database.")

        print(f"Data from {original_db} has been successfully saved to separate tables in the main databases.")

    finally:
        conn.close()
        if os.path.exists(processing_db):
            os.remove(processing_db)
            print(f"Processing completed, temporary database file {processing_db} has been deleted \n")

def calculate_and_save_averages(compute_conn):
    cursor = compute_conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Create or open a SQLite file to store intermediate results
    results_conn = sqlite3.connect('./res/results.sqlite')

    # Group tables by device ID
    device_groups = {}
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        match = re.search(r'device_(\d+)', table_name)
        if match:
            device_id = match.group(1)
            if device_id not in device_groups:
                device_groups[device_id] = []
            device_groups[device_id].append(table_name)

    for device_id, device_tables in device_groups.items():
        for table_name in device_tables:

            if 'overlap_False' in table_name:
                # Calculate ground_truth (average duration for non-overlap tables)
                query = f'''
                SELECT kShortName, kDemangledName, deviceId, AVG(duration) AS ground_truth
                FROM {table_name}
                GROUP BY kShortName, kDemangledName, deviceId
                '''
                df_ground_truth = pd.read_sql_query(query, compute_conn)
                df_ground_truth.to_sql(f'ground_truth_device_{device_id}', results_conn, if_exists='append', index=False)

            elif 'overlap_True' in table_name:
                # Calculate average duration and overlap rate for overlap tables
                query = f'''
                SELECT kShortName, kDemangledName, deviceId, 
                       SUM(duration) AS total_duration,
                       SUM(overlap_time) AS total_overlap_time,
                       AVG(duration) AS avg_duration
                FROM {table_name}
                WHERE overlap_ratio > 0.0
                GROUP BY kShortName, kDemangledName, deviceId
                '''
                df_overlap = pd.read_sql_query(query, compute_conn)
                df_overlap['overlap_rate'] = df_overlap['total_overlap_time'] / df_overlap['total_duration']
                df_overlap.to_sql(f'overlap_device_{device_id}', results_conn, if_exists='append', index=False)

        # Join the two tables using SQL with a LEFT JOIN to ensure all deviceIds are included
        combined_query = f'''
        SELECT gt.kShortName, gt.kDemangledName, gt.deviceId, gt.ground_truth, 
               ol.avg_duration, ol.overlap_rate
        FROM ground_truth_device_{device_id} AS gt
        LEFT JOIN overlap_device_{device_id} AS ol
        ON gt.kShortName = ol.kShortName AND 
           gt.kDemangledName = ol.kDemangledName AND 
           gt.deviceId = ol.deviceId
        '''

        combined_df = pd.read_sql_query(combined_query, results_conn)

        # Ensure that there are no divisions by zero
        combined_df = combined_df[(combined_df['overlap_rate'] > 0) & (combined_df['ground_truth'] > 0)]

        # Calculate slowdown and append it to the DataFrame
        combined_df['slowdown'] = (combined_df['avg_duration'] - (1 - combined_df['overlap_rate']) * combined_df['ground_truth']) / (combined_df['overlap_rate'] * combined_df['ground_truth']) - 1

        # Handle any potential NaN or infinite values in slowdown
        # combined_df['slowdown'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        combined_df['slowdown'] = combined_df['slowdown'].replace([float('inf'), -float('inf')], float('nan'))

        # Save the final combined data to a separate Excel file for each device
        combined_df.to_excel(f'slowdown_device_{device_id}.xlsx', index=False)
        print(f"Combined durations with slowdown have been saved to 'slowdown_device_{device_id}.xlsx'")

    # Close the intermediate results connection
    results_conn.close()

def main():
    prepare_res_directory()
    
    sqlite_files = [f for f in os.listdir('.') if f.endswith('.sqlite')]

    comm_conn = sqlite3.connect('./res/comm_results.sqlite')
    compute_conn = sqlite3.connect('./res/compute_results.sqlite')

    for sqlite_file in sqlite_files:
        process_sqlite_file(sqlite_file, comm_conn, compute_conn)

    calculate_and_save_averages(compute_conn)

    comm_conn.close()
    compute_conn.close()

if __name__ == '__main__':
    main()
