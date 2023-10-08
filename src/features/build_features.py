import pandas as pd
from IPython.display import display  

def inspect_dataframe(df, df_name=None):
    """
    Display various data inspection outputs for a given DataFrame.
    
    Parameters:
    - df : DataFrame
        The DataFrame to inspect.
    - df_name : str, optional
        The name of the DataFrame to display as a header. Default is None.
    """
    
    if df_name:
        print(f"\nData Inspection for DataFrame: {df_name}\n" + "-"*50)
    
    # Basic info about the dataframe
    print("Basic Info:")
    df.info()
    
    # Display the first few rows of the dataframe
    print("\nHead (first few rows):")
    display(df.head())
    
    # Print the shape of the dataframe
    print('\nShape of df:', df.shape)
    
    # Display summary statistics for numerical columns
    print("\nSummary Statistics:")
    display(df.describe())
    
    # Count of missing values for each column
    print("\nMissing Values Count:")
    display(df.isnull().sum())
    
    # Unique values count for each column
    print("\nUnique Values Count:")
    display(df.nunique())
    
    # Adding a separator for clarity
    print("\n" + "="*50 + "\n")

    


def preprocess_calendar(calendar_df, calendar_events_df):
    # Create a date range from 2011-01-29 to 2016-06-19
    date_range = pd.date_range(start='2011-01-29', end='2016-06-19')
    continuous_df = pd.DataFrame({'date': date_range})

    # Convert the 'date' column in calendar_events_df to datetime data type
    calendar_events_df['date'] = pd.to_datetime(calendar_events_df['date'])

    # Merge to fill missing dates and set 'no specific event' in missing event columns
    df3 = continuous_df.merge(calendar_events_df, on='date', how='left')
    df3['event_name'].fillna('no specific event', inplace=True)
    df3['event_type'].fillna('none', inplace=True)

    # Group by date to handle any duplicates
    calendar_events = df3.groupby('date').agg({'event_name': ' & '.join, 'event_type': ' & '.join}).reset_index()

    # Merge the two calendar dataframes
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    final_calendar = pd.merge(calendar_df, calendar_events, on="date", how="inner")
    final_calendar.rename(columns={"d": "Sales Day"}, inplace=True)
    return final_calendar

def filter_train_data(train, state_id='CA', store_id='CA_1', cat_id='FOODS', dept_id='FOODS_1', item_id='FOODS_1_001'):
    filtered_data = train[(train['state_id'] == state_id) &
                          (train['store_id'] == store_id) &
                          (train['cat_id'] == cat_id) &
                          (train['dept_id'] == dept_id) &
                          (train['item_id'] == item_id)]
    return filtered_data

def process_sales_data(train, calendar_df, sell_prices):
    sales_train = train.drop(train.columns[1:6], axis=1)
    sales_train = pd.melt(sales_train, id_vars=['id'], var_name='Sales Day', value_name='Sales')
    
    sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_evaluation"
    sell_prices = sell_prices[["id", "wm_yr_wk", "sell_price"]]
    
    Sales = pd.merge(sales_train, calendar_df, on=['Sales Day'], how='inner')
    Sales = pd.merge(Sales, sell_prices, on=['id', 'wm_yr_wk'], how='inner')
    Sales.rename(columns={'sell_price': 'whole_week_selling_price'}, inplace=True)
    Sales['Sales Day'] = Sales['Sales Day'].str.replace('d_', '').astype(int)
    Sales.rename(columns={'Sales': 'Daily volume of sales'}, inplace=True)
    Sales['Weekly volume of sales'] = Sales.groupby('wm_yr_wk')['Daily volume of sales'].transform('sum')
    Sales['Daily selling price'] = Sales.apply(lambda row: 0 if row['Weekly volume of sales'] == 0 else 
                                               (row['whole_week_selling_price'] / row['Weekly volume of sales']) * 
                                               row['Daily volume of sales'], axis=1)
    columns_to_keep = ['date', 'Sales Day', 'event_name', 'event_type', 'Daily volume of sales', 'Daily selling price']
    Sales = Sales[columns_to_keep]
    Sales['Sales Revenue'] = Sales['Daily volume of sales'] * Sales['Daily selling price']
    return Sales
