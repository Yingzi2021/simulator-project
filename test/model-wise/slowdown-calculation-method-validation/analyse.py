import sqlite3
import shutil
import os
import pandas as pd
import numpy as np
import re
# slowdown approximate method validation
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
        # Add new columns cudaAPIName, kShortName, kDemangledName, and duration
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN cudaAPIName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kShortName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN kDemangledName TEXT;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN duration INTEGER;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_time INTEGER DEFAULT 0;")
        cursor.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN overlap_ratio REAL DEFAULT 0;")

        # Update cudaAPIName column
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET cudaAPIName = (
            SELECT value FROM StringIds
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
            ON r.nameId = StringIds.id
            AND CUPTI_ACTIVITY_KIND_KERNEL.correlationId = r.correlationId);
        ''')

        # Update kShortName column
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kShortName = (
            SELECT value FROM StringIds WHERE shortName = StringIds.id);
        ''')

        # Update kDemangledName column
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET kDemangledName = (
            SELECT value FROM StringIds WHERE demangledName = StringIds.id);
        ''')

        # Calculate and update the duration column using SQL
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET duration = end - start;
        ''')

        # Calculate overlap time using SQL and update the overlap_time column
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

        # Calculate overlap ratio and update the overlap_ratio column
        cursor.execute('''
        UPDATE CUPTI_ACTIVITY_KIND_KERNEL
        SET overlap_ratio = CAST(overlap_time AS REAL) / duration
        WHERE duration > 0;
        ''')

        # Commit all changes at once
        conn.commit()

        # Extract the base name of the original database file without extension
        base_name = os.path.splitext(os.path.basename(original_db))[0]

        # Perform the first query for 'comm' data
        query_nccl = '''
        SELECT cudaAPIName, kShortName, kDemangledName, deviceId, start, end, duration, dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE kShortName IS NOT NULL 
            AND kShortName LIKE '%nccl%'
            AND kDemangledName IS NOT NULL
            AND cudaAPIName IS NOT NULL
        ORDER BY start ASC
        '''
        
        df_nccl = pd.read_sql_query(query_nccl, conn)

        # Add an 'id' column based on the row number (starting from 1)
        df_nccl['id'] = range(1, len(df_nccl) + 1)

        # Move 'id' to be the first column
        cols = ['id'] + [col for col in df_nccl.columns if col != 'id']
        df_nccl = df_nccl[cols]

        # Save the 'comm' data to a new table in the 'comm_results.sqlite' database
        df_nccl.to_sql(base_name, comm_conn, if_exists='replace', index=False)

        # Perform the second query for 'compute' data
        query_other = '''
        SELECT cudaAPIName, kShortName, kDemangledName, deviceId, start, end, duration, dynamicSharedMemory, overlap_time, overlap_ratio
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE kShortName IS NOT NULL 
            AND kShortName NOT LIKE '%nccl%'
            AND kDemangledName IS NOT NULL
            AND cudaAPIName IS NOT NULL
        ORDER BY deviceId ASC, kDemangledName ASC, start ASC
        '''

        df_other = pd.read_sql_query(query_other, conn)

        # Split the DataFrame by deviceId and process each group
        grouped = df_other.groupby('deviceId')
        for device_id, group in grouped:
            # Add an 'id' column based on the row number (starting from 1)
            group['id'] = range(1, len(group) + 1)

            # Move 'id' to be the first column
            cols = ['id'] + [col for col in group.columns if col != 'id']
            group = group[cols]

            # Save each group to a new table in the 'compute_results.sqlite' database
            table_name = f"{base_name}_device_{device_id}"
            group.to_sql(table_name, compute_conn, if_exists='replace', index=False)

            print(f"Data for device {device_id} has been saved to table {table_name} in the compute database.")

        print(f"Data from {original_db} has been successfully saved to separate tables in the main databases.")

    finally:
        # Close the database connection
        conn.close()

        # Delete the processing database file
        if os.path.exists(processing_db):
            os.remove(processing_db)
            print(f"Processing completed, temporary database file {processing_db} has been deleted \n")


# Function to calculate duration slowdown between {name}-ws1 and {name}-ws{n}-d{n}
def calculate_slowdown(compute_conn, slowdown_conn):
    cursor = compute_conn.cursor()

    # Get all table names from the compute database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Regular expression to match table names and extract the base name
    ws1_pattern = re.compile(r'^(.*)_ws1_device_\d+$')  # Matches tables ending in "_ws1_device_{n}"
    other_pattern = re.compile(r'^(.*)_ws\d+_device_\d+$')  # Matches tables like "_ws{n}_device_{n}"

    # Group tables by base name
    base_tables = {}
    for table in tables:
        name = table[0]
        ws1_match = ws1_pattern.match(name)
        other_match = other_pattern.match(name)

        if ws1_match:
            base_name = ws1_match.group(1)
            base_tables[base_name] = {'ws1': name, 'others': []}
        elif other_match:
            base_name = other_match.group(1)
            if base_name in base_tables:
                base_tables[base_name]['others'].append(name)
            else:
                # If we find an "other" table without a matching ws1 table, we should create an entry
                base_tables[base_name] = {'ws1': None, 'others': [name]}
    
    # Check for cases where ws1 is missing
    for base_name, table_group in base_tables.items():
        if not table_group['ws1']:
            print(f"Warning: No matching ws1 table for base name: {base_name}")
    
    # Calculate slowdown for each base name
    for base_name, table_group in base_tables.items():
        ws1_table = table_group['ws1']
        other_tables = table_group['others']

        if not ws1_table:
            continue  # Skip this base_name if there's no ws1 table
        
        for other_table in other_tables:
            slowdown_table = f"{other_table}_slowdown"

            # SQL query to calculate the slowdown
            calculate_slowdown_sql = f"""
            SELECT
                t1.id,
                t1.cudaAPIName,
                t1.kShortName,
                t1.kDemangledName,
                t1.duration AS duration_ws1,  -- Duration from ws1 table(ground truth)
                t2.duration,                  
                t2.overlap_ratio,             
                ((CAST(t2.duration AS REAL) - CAST(t1.duration AS REAL)) / CAST(t1.duration AS REAL)) AS slowdown
            FROM {ws1_table} t1
            JOIN {other_table} t2 ON t1.id = t2.id
            WHERE t1.kDemangledName = t2.kDemangledName;
            """

            # Use pandas to run the SQL query and store the result in a DataFrame
            df_slowdown = pd.read_sql_query(calculate_slowdown_sql, compute_conn)

            # Save the DataFrame to the slowdown database
            df_slowdown.to_sql(slowdown_table, slowdown_conn, if_exists='replace', index=False)

            print(f"Slowdown table {slowdown_table} created in slowdown_results.sqlite.")

    # Commit changes to the slowdown database
    slowdown_conn.commit()


def calculate_average_slowdown(slowdown_conn, output_directory):
    cursor = slowdown_conn.cursor()

    # Get all slowdown table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Process each slowdown table
    for table in tables:
        table_name = table[0]

        # Extract the device number using regex
        match = re.search(r'device_(\d+)', table_name)
        if not match:
            print(f"Could not extract device number from {table_name}, skipping...")
            continue

        device_num = match.group(1)

        # SQL query to calculate both the average slowdown for overlap_ratio == 0.0 and 1.0
        query = f"""
        SELECT 
            kShortName, 
            kDemangledName,
            COALESCE(AVG(CASE WHEN overlap_ratio = 0.0 THEN slowdown END), 'NaN') AS error,  -- Average slowdown for overlap_ratio == 0.0
            COALESCE(AVG(CASE WHEN overlap_ratio = 1.0 THEN slowdown END), 'NaN') AS slowdown  -- Average slowdown for overlap_ratio == 1.0
        FROM {table_name}
        GROUP BY kShortName, kDemangledName;
        """

        df = pd.read_sql_query(query, slowdown_conn)

        if df.empty:
            print(f"No relevant data found in table {table_name}, skipping...")
            continue

        # Define the output CSV path for this device
        output_csv_path = os.path.join(output_directory, f"average_slowdown_device_{device_num}.csv")

        # Save the DataFrame to the corresponding CSV file for this device
        df.to_csv(output_csv_path, index=False)
        print(f"Average slowdown for {table_name} (device {device_num}) has been calculated and saved to {output_csv_path}")


def calculate_partial_overlap_averages(compute_conn, output_directory):
    cursor = compute_conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [table[0] for table in cursor.fetchall()]

    # Identify all ws1 tables
    ws1_tables = {table: table for table in all_tables if "ws1" in table}

    # Process each table, excluding the ones with "ws1" in their name for main processing
    for table in all_tables:
        if "ws1" not in table:
            # Extract the device number from the ws{n} table name
            match = re.search(r'device_(\d+)', table)
            if not match:
                print(f"Could not extract device number from {table}, skipping...")
                continue

            device_num = match.group(1)

            # Find the corresponding ws1 table with the same device number
            ws1_table = next((ws1_table for ws1_table in ws1_tables if f"device_{device_num}" in ws1_table), None)

            if not ws1_table:
                print(f"Warning: Corresponding ws1 table not found for {table}. Skipping...")
                continue

            # SQL query to calculate the average duration and average overlap rate for rows
            # where overlap_ratio is between 0.0 and 1.0, grouped by kShortName and kDemangledName
            query = f"""
            SELECT 
                kShortName,
                kDemangledName,
                SUM(overlap_time) AS total_overlap_time,
                SUM(duration) AS total_duration,
                AVG(duration) AS average_duration,
                (CAST(SUM(overlap_time) AS REAL) / CAST(SUM(duration) AS REAL)) AS average_overlap_rate
            FROM {table}
            WHERE overlap_ratio > 0.0 AND overlap_ratio < 1.0
            GROUP BY kShortName, kDemangledName;
            """
            
            # Execute the query and load the results into a DataFrame
            df = pd.read_sql_query(query, compute_conn)

            # Now, calculate the ground truth from the corresponding ws1 table
            ground_truth_query = f"""
            SELECT 
                kShortName,
                kDemangledName,
                AVG(duration) AS ground_truth
            FROM {ws1_table}
            GROUP BY kShortName, kDemangledName;
            """

            # Execute the ground truth query
            ground_truth_df = pd.read_sql_query(ground_truth_query, compute_conn)

            # Merge the ground truth data with the main DataFrame
            df = pd.merge(df, ground_truth_df, on=['kShortName', 'kDemangledName'], how='left')

            # Define the output CSV path for this device
            output_csv_path = os.path.join(output_directory, f"partial_overlap_device_{device_num}.csv")

            # Save the DataFrame to the corresponding CSV file for this device
            if not df.empty:
                df.to_csv(output_csv_path, index=False)
                print(f"Data for device {device_num} has been processed and saved to {output_csv_path}")

def merge_slowdown_calculate_duration_and_error(output_directory):
    # Get all files in the output directory
    files = os.listdir(output_directory)
    
    # Filter for partial_overlap and average_slowdown files
    partial_overlap_files = {f for f in files if f.startswith('partial_overlap_device') and f.endswith('.csv')}
    average_slowdown_files = {f for f in files if f.startswith('average_slowdown_device') and f.endswith('.csv')}
    
    # Process each pair of partial_overlap and average_slowdown files
    for partial_file in partial_overlap_files:
        # Extract device number from the file name
        device_num = partial_file.split('device_')[1].split('.csv')[0]
        
        # Construct the corresponding average_slowdown file name
        average_file = f"average_slowdown_device_{device_num}.csv"
        
        if average_file in average_slowdown_files:
            # Load the partial_overlap and average_slowdown data
            partial_df = pd.read_csv(os.path.join(output_directory, partial_file))
            average_df = pd.read_csv(os.path.join(output_directory, average_file))
            
            # Merge the dataframes on <kShortName, kDemangledName>
            merged_df = pd.merge(partial_df, average_df[['kShortName', 'kDemangledName', 'slowdown']],
                                 on=['kShortName', 'kDemangledName'], how='left')
            
            # Calculate the calculated_duration column
            merged_df['calculated_duration'] = (
                merged_df['ground_truth'] * (1 - merged_df['average_overlap_rate']) +
                merged_df['ground_truth'] * merged_df['average_overlap_rate'] * (1 + merged_df['slowdown'])
            )
            
            # Calculate the error column
            merged_df['error'] = (merged_df['average_duration'] - merged_df['calculated_duration']) / merged_df['average_duration']
            
            # Save the updated partial_overlap file back to disk
            merged_df.to_csv(os.path.join(output_directory, partial_file), index=False)
            
            print(f"Calculated duration and error added to {partial_file} from {average_file}")
        else:
            print(f"Warning: No matching average_slowdown file for {partial_file}")


# Main function to find and process all .sqlite files in the current directory
def main():
    # Ensure the result directory exists
    result_dir = './result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    # Search for all .sqlite files in the current directory
    sqlite_files = [f for f in os.listdir('.') if f.endswith('.sqlite')]

    # Create or open the main SQLite databases for comm, compute, and slowdown data
    comm_conn = sqlite3.connect('./result/comm_results.sqlite')
    compute_conn = sqlite3.connect('./result/compute_results.sqlite')
    slowdown_db = sqlite3.connect('./result/slowdown_results.sqlite')

    # Process each .sqlite file found
    for sqlite_file in sqlite_files:
        process_sqlite_file(sqlite_file, comm_conn, compute_conn)

    # Calculate the duration slowdown between tables in compute.sqlite and save to slowdown_results.sqlite
    calculate_slowdown(compute_conn, slowdown_db)

    # Calculate the average slowdown and save to CSV using SQL
    calculate_average_slowdown(slowdown_db, './result/')

    # Calculate the average duration and overlap rate for partial overlaps and save to CSV using SQL
    calculate_partial_overlap_averages(compute_conn, './result/')

    merge_slowdown_calculate_duration_and_error('./result/')

    # Close the main database connections
    comm_conn.close()
    compute_conn.close()
    slowdown_db.close()

if __name__ == '__main__':
    main()
