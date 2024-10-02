import sqlite3
import shutil
import os
import pandas as pd
import re
import openpyxl

def prepare_res_directory():
    res_dir = './res'
    if os.path.exists(res_dir):
        # Clean the directory by deleting all files
        shutil.rmtree(res_dir)
    # Create the directory
    os.makedirs(res_dir)
    
def save_grouped_data_to_db(df, base_name, suffix, compute_conn):
    # Group by deviceId and save each group to a separate table
    grouped = df.groupby('deviceId')

    for device_id, group in grouped:
        group['id'] = range(1, len(group) + 1)
        cols = ['id'] + [col for col in group.columns if col != 'id']
        group = group[cols]

        table_name = f"{base_name}_device_{device_id}_{suffix}"
        group.to_sql(table_name, compute_conn, if_exists='replace', index=False)

        print(f"Data for device {device_id} has been saved to table {table_name} in the compute database.")

def retrieve_table_names(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # Flatten the result and return table names as a list
    return [table[0] for table in tables]

def calculate_average_slowdown(compute_conn, avg_memcpy_tables, memcpy_tables):
    all_slowdowns = []
    
    for avg_table, true_table in zip(avg_memcpy_tables, memcpy_tables):
        # Retrieve ground truth from avg_memcpy table
        avg_memcpy_df = pd.read_sql_query(f'''
        SELECT deviceId, avg_duration, copyKind 
        FROM {avg_table};
        ''', compute_conn)

        # Retrieve data from memcpy table with overlap information
        df_memcpy = pd.read_sql_query(f'''
        SELECT deviceId, duration, overlap_ratio, copyKind 
        FROM {true_table}
        WHERE overlap_ratio != 0
        ''', compute_conn)

        # Ensure we match by deviceId
        slowdown_df = pd.merge(df_memcpy, avg_memcpy_df, on=['deviceId', 'copyKind'])

        # Calculate slowdown
        slowdown_df['slowdown'] = (slowdown_df['duration'] - slowdown_df['avg_duration']) / (slowdown_df['overlap_ratio'] * slowdown_df['avg_duration'])

        # Average slowdown per device
        avg_slowdown = slowdown_df.groupby('deviceId')['slowdown'].mean().reset_index()

        # Add to results
        all_slowdowns.append(avg_slowdown)

    return pd.concat(all_slowdowns, ignore_index=True)

def process_sqlite_file(original_db, comm_conn, compute_conn):
    # Generate processing database filename
    processing_db = 'processing-db.sqlite'

    # Backup the database (copy the original database file to a new file)
    shutil.copyfile(original_db, processing_db)
    print(f"Database {original_db} has been backed up as {processing_db}")

    # Extract the base name of the original database file without extension
    base_name = os.path.splitext(os.path.basename(original_db))[0]

    # Connect to the processing SQLite database
    conn = sqlite3.connect(processing_db)
    cursor = conn.cursor()

    try:
        # Add new columns to MEMCPY table if they don't exist
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN duration INTEGER;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN overlap_time INTEGER DEFAULT 0;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN overlap_ratio REAL DEFAULT 0;")

        # Calculate duration (end - start)
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_MEMCPY
        SET duration = end - start;
        ''')

        # Processing overlap_False case
        if "overlap_False" in base_name:
            # Calculate average duration grouped by deviceId and copyKind
            cursor.execute('''
            SELECT deviceId, AVG(duration) AS avg_duration, copyKind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY deviceId, copyKind;
            ''')

            avg_durations = cursor.fetchall()

            # Convert the result into a pandas DataFrame
            avg_df = pd.DataFrame(avg_durations, columns=["deviceId", "avg_duration", "copyKind"])

            # Save grouped data to the database
            save_grouped_data_to_db(avg_df, base_name, 'avg_memcpy', compute_conn)

        # Processing overlap_True case
        elif "overlap_True" in base_name:
            # Ensure necessary columns exist and populate them via a join with StringIds
            cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN duration INTEGER;")

            # Calculate duration for CUPTI_ACTIVITY_KIND_KERNEL
            cursor.execute('''
            UPDATE CUPTI_ACTIVITY_KIND_KERNEL
            SET duration = end - start;
            ''')

            # Calculate overlap time using join with StringIds to identify communication operations (e.g., NCCL)
            cursor.execute('''
            WITH Overlaps AS (
                SELECT 
                    m.deviceId AS device_id,
                    m.start AS memcpy_start,
                    m.end AS memcpy_end,
                    SUM(
                        CASE
                            WHEN m.start < c.end AND m.end > c.start THEN
                                MIN(m.end, c.end) - MAX(m.start, c.start)
                            ELSE 0
                        END
                    ) AS total_overlap_time
                FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                JOIN CUPTI_ACTIVITY_KIND_KERNEL c
                ON m.deviceId = c.deviceId
                JOIN StringIds s
                ON c.shortName = s.id
                AND s.value LIKE '%nccl%'
                GROUP BY m.deviceId, m.start, m.end
            )
            UPDATE CUPTI_ACTIVITY_KIND_MEMCPY
            SET overlap_time = (
                SELECT total_overlap_time 
                FROM Overlaps 
                WHERE Overlaps.device_id = CUPTI_ACTIVITY_KIND_MEMCPY.deviceId
                AND Overlaps.memcpy_start = CUPTI_ACTIVITY_KIND_MEMCPY.start
                AND Overlaps.memcpy_end = CUPTI_ACTIVITY_KIND_MEMCPY.end
            );
            ''')

            # Calculate overlap ratio (overlap_time / duration)
            cursor.execute('''
            UPDATE CUPTI_ACTIVITY_KIND_MEMCPY
            SET overlap_ratio = CAST(overlap_time AS REAL) / duration
            WHERE duration > 0;
            ''')

            conn.commit()

            # Query the necessary columns from MEMCPY table
            query_memcpy = '''
            SELECT start, end, deviceId, bytes, copyKind, duration, overlap_time, overlap_ratio
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE start IS NOT NULL AND end IS NOT NULL;
            '''

            df_memcpy = pd.read_sql_query(query_memcpy, conn)

            # Save grouped data to the database
            save_grouped_data_to_db(df_memcpy, base_name, 'memcpy', compute_conn)

        print(f"Memcpy data from {original_db} has been successfully saved to separate tables in the main databases.")

    finally:
        conn.close()
        if os.path.exists(processing_db):
            os.remove(processing_db)
            print(f"Processing completed, temporary database file {processing_db} has been deleted \n")

def main():
    prepare_res_directory()
    
    sqlite_files = [f for f in os.listdir('.') if f.endswith('.sqlite')]

    comm_conn = sqlite3.connect('./res/comm_results.sqlite')
    compute_conn = sqlite3.connect('./res/compute_results.sqlite')

    for sqlite_file in sqlite_files:
        process_sqlite_file(sqlite_file, comm_conn, compute_conn)

    # Retrieve all table names from compute_results.sqlite
    compute_tables = retrieve_table_names(compute_conn)

    # Find avg_memcpy and memcpy tables
    avg_memcpy_tables = [table for table in compute_tables if "avg_memcpy" in table]
    memcpy_tables = [table for table in compute_tables if "memcpy" in table and "overlap_True" in table]

    # Calculate average slowdown for each device
    avg_slowdown_df = calculate_average_slowdown(compute_conn, avg_memcpy_tables, memcpy_tables)

    # Save the average slowdown results to a CSV
    avg_slowdown_df.to_excel('./res/memcpy_average_slowdown.xlsx', index=False)

    print("Average slowdown per device has been saved to './res/average_slowdown_per_device.csv'.")

    comm_conn.close()
    compute_conn.close()

if __name__ == '__main__':
    main()
