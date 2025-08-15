import logging
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

# Define Constants
pathToData = 'data'
masterName = 'master.csv'
dneName = 'dne.csv'

# Configure logging
def setup_logging():
    logger = logging.getLogger('analyzeScrape')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File Handler
    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    
    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create logger instance
logger = setup_logging()

def relocateLogger(runName, pathToData = pathToData):
    new_log_path = f'{pathToData}/{runName}/log.txt'
    shutil.move('log.txt', new_log_path)

def processData(runName, pathToData = pathToData, masterName = masterName, dneName = dneName):
    # Change directory to where file exists
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read Data
    try:
        master = pd.read_csv(f'{pathToData}/{runName}/{masterName}')
        dne = pd.read_csv(f'{pathToData}/{runName}/{dneName}')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}. Check pathToData/runName and filenames.")
        raise

    # Create alphaNumeric column
    master = enhanceAlphaNumeric(master, 'VC Firm')

    # Calculate Initial Counts for DNE
    initialCountsDNE = master.groupby('Analyst').count()

    # Process DNE
    dne = enhanceAlphaNumeric(dne, 'Company Name')
    dne = dne['AlphaNumeric'].to_list()
    master = master[master['AlphaNumeric'].isin(dne) == False].sort_values('VC Firm')

    # Calculate Final Counts for DNE
    finalCountsDNE = master.groupby('Analyst').count()
    calculateCompliace(initialCountsDNE, finalCountsDNE, 'Do Not Email Compliance')

    # Ensure everything is an email
    initialCountsInfo = master.groupby('Analyst').count()
    epat = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    master['isEmail'] = master['Email'].str.fullmatch(epat)
    master = master[master['isEmail'] == True]

    # Drop missing data
    master.dropna(subset=["Name", "Email" , 'VC Firm'], inplace=True)

    # Info Compliance
    finalCountsInfo = master.groupby('Analyst').count()
    countsByAnalyst = calculateCompliace(initialCountsInfo, finalCountsInfo, 'Provided Information Compliance')
    countsByAnalyst[['Analyst' , 'finalCount']].to_csv(f'{pathToData}/{runName}/counts.csv', index = False)

    # Drop unnessary columns
    master.drop('isEmail', axis=1)

    # Analyze duplicates
    duplicates = analyzeDuplicates(master)
    duplicates.to_csv(f'{pathToData}/{runName}/duplicates.csv', index = False)

    # Drop Duplicates
    colsToCheck = ['AlphaNumeric' , 'Name', 'Email']
    masterNoDuplicates = master.copy()
    for col in colsToCheck:
        masterNoDuplicates.drop_duplicates(subset = col, keep = 'first', inplace = True)
    masterNoDuplicates = masterNoDuplicates.sort_values('VC Firm')

    # Save output
    masterNoDuplicates.to_csv(f'{pathToData}/{runName}/output.csv', index = False)

    return masterNoDuplicates

def gatherDuplicates(df):
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.reset_index().rename(columns = {'index' : 'ID'})
    cols = ['AlphaNumeric' , 'Name' , 'Email']
    duplicates = pd.DataFrame()

    for col in cols:
        duplicates = pd.concat([duplicates, df[df.duplicated(subset=[col], keep=False)].sort_values(col)], axis = 0)

    duplicates = duplicates.drop_duplicates()
    duplicates = duplicates.drop(columns = 'ID').reset_index(drop = True)

    return duplicates

def gatherExactDuplicates(df):
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)    
    subset_cols = ['VC Firm', 'HQ Location', 'Name', 'Position', 'Email', 'Notes']
    duplicates = df[df.duplicated(subset=subset_cols, keep=False)].sort_values(subset_cols).copy()
    return duplicates

def computeDuplicatesMatrix(df):
    if 'RowID' not in df.columns:
        df['RowID'] = range(len(df))

    analysts = df['Analyst'].unique()
    matrix = pd.DataFrame(0, index=analysts, columns=analysts)
    
    cols = ['AlphaNumeric', 'Name', 'Email']

    # Keep track of counted row pairs to avoid double-counting
    counted_pairs = set()

    # For each duplicate column, create a mapping of value -> analysts
    for col in cols:
        # Convert to list of lists instead of tuples for safer handling
        grouped = df.dropna(subset=[col]).groupby(col)[['Analyst', 'RowID']].apply(lambda x: x.values.tolist())
        
        for group in grouped:
            if not group:  # Skip if group is empty
                continue
            # group is now a list of lists: [Analyst, RowID]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    try:
                        a1, r1 = group[i]
                        a2, r2 = group[j]
                        # Create a unique identifier for the row pair (order independent)
                        row_pair = tuple(sorted([r1, r2]))
                        if row_pair not in counted_pairs:
                            matrix.at[a1, a2] += 1
                            matrix.at[a2, a1] += 1
                            counted_pairs.add(row_pair)
                    except (ValueError, IndexError):
                        continue  # Skip malformed entries

    matrix = matrix.replace(0, '')
    return matrix

def analyzeDuplicates(df):
    duplicates = gatherDuplicates(df)
    duplicatesMatrix = computeDuplicatesMatrix(duplicates)

    logger.info(f"\n\nDuplicates\n{'-'*100}")
    logger.info(f"\n{duplicatesMatrix}")

    exactDuplicates = gatherExactDuplicates(df)

    exactDuplicates.to_csv('debug.csv')
    ExactDuplicatesMatrix = computeDuplicatesMatrix(exactDuplicates)

    logger.info(f"\n\nExact Duplicates\n{'-'*100}")
    logger.info(f"\n{ExactDuplicatesMatrix}")

    return duplicates

def enhanceAlphaNumeric(df, colName):
    pattern = re.compile(r'[^A-Za-z0-9]|inc|vc|corp', re.IGNORECASE)
    df['AlphaNumeric'] = df[colName].str.lower().str.replace(pattern, '', regex = True)
    return df

def calculateCompliace(initialCounts, finalCounts, name):
    finalCounts = finalCounts.reset_index()[['Analyst' , 'Name']].rename(columns = {'Name': 'finalCount'})
    initialCounts = initialCounts.reset_index()[['Analyst' , 'Name']].rename(columns = {'Name': 'initialCount'})
    comboCounts = pd.merge(left = initialCounts, right = finalCounts, on = 'Analyst', how = 'outer')
    comboCounts['compliance'] = comboCounts['finalCount'] / comboCounts['initialCount']
    comboCounts = comboCounts.sort_values('compliance').reset_index(drop = True)
    logger.info(f"\n\n{name}\n{'-'*100}")
    logger.info(f"\n{comboCounts[comboCounts['compliance'] < 1]}")
    return comboCounts

def main():
    # Change directory to where file exists
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Get user arguments
    args = sys.argv[1:]
    num_args = len(args)
    if not (0 < num_args < 5):
        logger.error(f"Error: Expected 1-4 arguments, got {num_args}")
        sys.exit(1)
    
    logger.info('\n')   

    logger.info(f'Arguments: {args}')

    # Process Data
    processData(*args)


    logger.info('\n')   

    # Move logger to correct folder
    relocateLogger(*args[:9])

if __name__ == '__main__':
    main()
