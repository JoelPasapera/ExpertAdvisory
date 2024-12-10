"""
ExpertAdvisory
"""

import polars as pl
import pathlib
from pathlib import Path, PurePath
import os
from datetime import datetime, time, timedelta
import pytz
from polars.exceptions import ComputeError
import torch
import torch.jit
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional
import plotly.graph_objects as go
import plotly.io as pio
from colorama import init, Fore, Back, Style


def setup_file_paths(path: str = None, show_paths: bool = False):
    """
    Summary:
        This function is used to execute the python script in the same directory where
        it is located or a custom one. This avoids problems in accessing and producing files.
    Args:
        path (str, optional): path where the python script will be executed. Defaults to None.
            Remember, the path must not contain the name of the file, only the name of the folder.
        show_paths (bool, optional): prints in console the paths.
            If True, the paths will be printed in the console. Defaults to False.
    Returns:
        directory_script (Path): the path or directory where this python file is located

    """
    # change the current working directory to the location of this script.
    if path is None:
        #  Directory where the script is located
        directory_script = Path(__file__).resolve().parent
    # change the current working directory to a custom location.
    else:
        #  Directory provided by the user
        directory_script = path
    # change directory
    os.chdir(str(directory_script))
    if show_paths is True:
        print(f"Current working directory: {directory_script}")
        print(f"Name of this script: {Path(__file__).name}")
    return directory_script


class CSVProcessor:
    """
    Summary:
    Class to process CSV files with a defined schema.

    Attributes:
        default_schema (dict): default schema for the data.
        schema (dict): Schema to be used to process the data.

    """

    def __init__(self, csv_name: str, schema: dict = None) -> None:
        # Define a name for csv data base
        self.csv_name = csv_name
        # Define the complete file name for csv data base
        self.csv_file_name = f"{csv_name}.csv"
        # Define the complete file name for csv main data base 1-minute
        self.csv_main_file_name = f"{csv_name}_1m.csv"
        # We define the default scheme (32 is cheaper than 64)
        self.default_schema = {
            "Date": pl.String,
            "Open": pl.Float32,
            "High": pl.Float32,
            "Low": pl.Float32,
            "Close": pl.Float32,
        }
        # Define the main data frame with all the history data
        self.df_1m = None
        # If no schema is provided, we use the default schema.
        if schema is not None and isinstance(schema, dict):
            self.schema = schema
        else:
            self.schema = self.default_schema
        # Verify
        self.check_and_create_csv()

    def validate_input(self):
        """
        Resume:
            Validate the input data.
        """
        # Check if the input data is valid

    def csv_processor(
        self,
        show: bool = False,
        return_tensor: bool = False,
        start_date: str = None,
        end_date: str = None,
        generate_csv_file: bool = False,
    ) -> tuple[pl.DataFrame, Optional[torch.Tensor]]:
        """Process a CSV file and optionally return a Polars DataFrame and a PyTorch tensor

        Args:
            show (bool, optional): Show the data frame. Defaults to False.
            return_tensor (bool, optional): return a tensor. Defaults to False.
            start_date (str, optional): the start date for the selection (format: "YYYY.MM.DD"). Defaults to None.
            end_date (str, optional): the end date for the selection (format: "YYYY.MM.DD"). Defaults to None.

        Returns:
            tuple[pl.DataFrame, torch.Tensor]: return a polars data frame and a tensor
        Example:
        >>> df, tensor = csv_processor(show=True, start_date="2019-01-7", end_date="2019-01-8")

        """
        try:

            # Load CSV as a LazyFrame
            df = pl.scan_csv(self.csv_file_name, has_header=False)

            # Convert str to datetime
            df = df.with_columns(
                [
                    pl.col("column_1")
                    .str.strptime(pl.Date, "%Y.%m.%d")
                    .alias("column_1"),
                    pl.col("column_2").str.strptime(pl.Time, "%H:%M").alias("column_2"),
                ]
            )
            df = df.with_columns(
                pl.col("column_1").dt.combine(pl.col("column_2")).alias("column_1")
            )

            # First, the column is interpreted as being in the America/Lima time zone (UTC-5).
            df = df.with_columns(
                pl.coalesce("column_1").dt.replace_time_zone("America/Lima")
            )

            # Second, convert the 'time' column to the New York time zone,
            # considering daylight saving time (UTC-4 in summer, UTC-5 in winter).
            df = df.with_columns(
                pl.col("column_1").dt.convert_time_zone("America/New_York")
            )

            # Finally, delete unnecessary column
            df = df.drop(["column_2", "column_7"])

            # rename columns
            df = df.rename(
                {
                    "column_1": "Date",
                    "column_3": "Open",
                    "column_4": "High",
                    "column_5": "Low",
                    "column_6": "Close",
                }
            )
            # convert the strings dates to right format
            if start_date is not None:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").astimezone(
                    pytz.timezone("America/New_York")
                )
            if end_date is not None:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").astimezone(
                    pytz.timezone("America/New_York")
                )
            # filter the table by start and end date
            if start_date or end_date is not None:
                df = df.filter(
                    (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date)
                )

            # materialize (convert lazy data frame to data frame)
            df = df.collect()
            if return_tensor is True:
                # Convert column 'dates' to a number (timestamp in nanoseconds)
                df_int = df.with_columns(
                    pl.col("Date").cast(
                        pl.Int64
                    )  # Switches to Int64 to get the timestamp
                )
                df_tensor = df_int.to_torch(return_type="tensor")
                torch.set_printoptions(sci_mode=False)

            else:
                df_tensor = None
            if show is True:
                print(df.shape)
                print(
                    "\nData frame:\n",
                    df,
                    "\nData frame converted to tensor:\n",
                    df_tensor,
                )
            if generate_csv_file is True:
                # Define the path of CSV file
                path = pathlib.Path(self.csv_main_file_name)
                # Delete the time information
                df = df.with_columns(
                    pl.col("Date").dt.to_string("%Y-%m-%d %H:%M:%S").alias("Date")
                )
                # Write the data frame to CSV file
                df.write_csv(path)

            return df, df_tensor
        except pl.exceptions.ComputeError:
            print("Error when making transformations")
            return None
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading CSV: {e}")
            return None

    def group_and_write_csv_by(self, time_frame: str | list):
        """_summary_

        Args:
            time_frame (str): time frame, like 5m, 10m, 15m, 1h, etc...

        Returns:
            pl.Dataframe: A new data frame
        Example:
        >>> df = roup_and_write_csv_by('1m') # 1-minute candlestick
        >>> df = roup_and_write_csv_by('1h') # 1-hour candlestick
        """

        # Grouping 1-minute candles to create x-minute candles and write it to a csv file
        def write_csv_time_frame():
            # transform 1-minute Japanese candlesticks to specified timeframe
            df_grouped = (
                self.df_1m.group_by(
                    pl.col("Date").dt.truncate(
                        time_frame
                    )  # rounds down the timestamp of each 1-minute candle
                    # to the start of the nearest 5-minute interval.
                )  # Group by each 5-minute interval
                .agg(
                    [
                        pl.first("Open").alias(
                            "Open"
                        ),  # Opening of the first sail of the group
                        pl.max("High").alias("High"),  # Maximum of the group
                        pl.min("Low").alias("Low"),  # Minimum of the group
                        pl.last("Close").alias(
                            "Close"
                        ),  # Closing of the last candle of the group
                    ]
                )
                .sort(
                    pl.col("Date").dt.truncate(time_frame)
                )  # Sort by truncated interval
            )

            # In order to avoid wrong formats in 'Date' column, time is converted to text.
            df_grouped = df_grouped.with_columns(
                pl.col("Date").dt.to_string("%Y-%m-%d %H:%M:%S").alias("Date")
            )
            # materialize or make physical the lazy frame
            df_grouped = df_grouped.collect()
            # create a csv file by writing the data frame
            df_grouped.write_csv(pathlib.Path(f"{self.csv_name}_{time_frame}.csv"))

        # if it is a str, then call once but if it a list call as needed
        if isinstance(time_frame, str):
            write_csv_time_frame()
        elif isinstance(time_frame, list):
            # iterate through the list of data frames to create csv files
            for tm in time_frame:
                # if exists, then pass
                if os.path.exists(f"{self.csv_name}_{tm}.csv"):
                    print(f"{self.csv_name}_{tm}.csv already exists, pass...")
                # if not exists, then create
                else:
                    self.group_and_write_csv_by(tm)
                    print(f"{self.csv_name}_{tm}.csv successfully created...")

    def check_and_create_csv(self) -> None:
        """Checks if the CSV file exists and creates it if not.
            Check if the CSV file already exists
        Args:
            time_frame (str, optional): Time frame for grouping, like '1m' or '1h'.
        """
        # If exists scan 1-minute candlestick csv file
        if os.path.exists(self.csv_main_file_name):
            pass
        # If not exists scan 1-minute candlestick csv file, create it and create common time frames
        else:
            # Step 1: Create 1-minute candlestick csv file
            self.df_1m, _ = self.csv_processor(
                show=False, return_tensor=False, generate_csv_file=True
            )
            # Step 2: Create common time frames (group by time frame and write principal csv files)
            self.group_and_write_csv_by(["5m", "10m", "15m", "30m", "1h", "4h", "1d"])

    def scan_csv(
        self, time_frame, materialize: bool = False
    ) -> pl.LazyFrame | pl.DataFrame:
        """Scans a CSV file and returns a LazyFrame or a DataFrame.
        Args:
            csv_file (str): Path to the CSV file.
            materialize (bool, optional): If True, returns a materialized DataFrame.
            Defaults to False.
        Returns:
            pl.LazyFrame | pl.DataFrame: A LazyFrame if materialize is False,
            or a DataFrame if materialize is True.
        """
        # file name that will be scan when the function is called
        csv_file_name = f"{self.csv_name}_{time_frame}.csv"

        def scan():
            # headers: Date,Open,High,Low,Close
            # schema == data type per each column, establish it for greater scanning efficiency
            # scan with a schema
            df = pl.scan_csv(
                csv_file_name, has_header=True, schema_overrides=self.schema
            )
            # convert text to time (data time)
            df = df.with_columns(
                pl.col("Date")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                .alias("Date")
            )
            return df

        if os.path.exists(csv_file_name):
            df = scan()
        else:
            self.group_and_write_csv_by(time_frame)
        if materialize is True:
            df = df.collect()
        return df


# This class is in charge of the exploratory analysis of the data (EDA).
class DataExplorer:
    """
    Resume:
        This class is in charge of the exploratory analysis of the data (EDA).
    """

    def __init__(self, df: pl.DataFrame):
        pass


# This class deals with the analysis of fundamental news.
class FinancialNewsAnalyzer:
    """
    Resume:
        This class deals with the analysis of fundamental news.
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df

    def web_scraping(self):
        """
        Resume:
            This method is in charge of the web scraping of the fundamental news.
        """
        # web scraping of news
        # return a new dataframe with the scraped news


# This class stores all trading strategies
class StrategyFactory:
    """
    Resume:
        This class stores all trading strategies.
    """

    def __init__(self):
        self.check_format()

    def check_format(self):
        """
        Resume:
            This method checks the input and the output format of the strategies.
        """
        # check if the format of the output is correct
        pass

    def strategy_1(
        self,
        df: pl.DataFrame,
        entry_time: str = "9:00",
        order: str = "buy",
        tp: int = 4,
        sl: int = 2,
        limit_time: str = "13:00:00",
        lot_size: int = 0.05,
    ) -> pl.DataFrame:
        """
        Resume:
            This strategy simply put an buy or sell in the same hour with same sl and tp.

        Args:
            df (pl.DataFrame): the dataframe to be used in strategy creation
            entry_time (str, optional): entry time order. Defaults to "8:00".
            order (str, optional): could buy or sell. Defaults to "buy".
            tp (int, optional): take profit. Defaults to 10.
            sl (int, optional): stop loss. Defaults to 2.
            limit_time (str, optional): max time to order life. Defaults to "13:00:00".
            lot_size (int, optional): lot. Defaults to 0.05.

        Raises:
            ValueError: error if invalid format is giving as an argument

        Returns:
            pl.DataFrame: the order parameters
        """
        # Avoid unused
        ny = f"{order}, {lot_size}"
        ny = len(ny) + lot_size
        # Get the time
        try:
            hours, minutes = map(int, entry_time.split(":"))
        except ValueError as exc:
            raise ValueError(f"The format of entry_time must be 'HH:MM'.{exc}") from exc
        # get data frame filtered by time
        df = df.filter(
            (pl.col("Date").dt.hour() == hours)
            & (pl.col("Date").dt.minute() == minutes)
        )
        # rename column
        df = df.rename(
            {
                "Date": "entry_time",
            }
        )
        # Get tp and sl
        df = df.with_columns(
            [
                (pl.col("High") + int(tp)).alias("tp"),
                (pl.col("Low") - int(sl)).alias("sl"),
            ]
        )

        # Delete unnecessary columns
        df = df.drop(["High", "Low", "Close"])

        # =========================================================================== #
        # This part indicates the time limit of the operation
        # (this part is extremely important to avoid excessive ram consumption in later filters).
        # =========================================================================== #

        hour, minute, second = limit_time.split(":")
        # create Date_max column
        df = df.with_columns(
            pl.datetime(
                pl.col("entry_time").dt.year(),
                pl.col("entry_time").dt.month(),
                pl.col("entry_time").dt.day(),
                int(hour),  # new hour
                int(minute),  # new minute
                int(second),  # new second
            ).alias("Date_max")
        )
        return df

    def strategy_2(self):
        """
        Resume:
            This strategy is based on Smart Money Concept.
            Use SMC to identify direction, location and execution.
        """

    # here you can create as many strategies as you want
    # ...


# This class is in charge of performing historical back testing.
class BackTesting:
    """
    Resume:
        This class is in charge of performing historical back testing.
    """

    def __init__(self):
        self.validate_format()

    def validate_format(self):
        """
        This method is used to validate the format of the data.
        """
        # Check if the input data is in the correct format

    def _get_max_min(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Resume:
            The columns 'tp' and 'sl' are obtained by finding
            the maximum and minimum between tp and sl per row.
        Args:
            df (pl.DataFrame): DataFrame with 'tp' and 'sl' columns.
            This is calculated from tp and sl.
        Returns:
            pl.Dataframe: The DataFrame with 'max' and 'min' columns.
        """
        # Get the maximum and minimum using CPU
        # =========================================================================== #
        # This part is to get max and min values, do not delete this logic, just optimize.
        # =========================================================================== #

        # Materialize lazy frame because lf doesn't have 'transpose' method but df does have it
        df = df.collect()

        # get max and min columns
        df = df.with_columns(
            df.select(["tp", "sl"])
            .transpose()
            .max()
            .transpose()
            .to_series()
            .alias("max"),
            df.select(["tp", "sl"])
            .transpose()
            .min()
            .transpose()
            .to_series()
            .alias("min"),
        )
        return df

    def _run_back_test(
        self, parameters: pl.DataFrame, df_1m: pl.DataFrame
    ) -> pl.DataFrame:
        """_summary_
        This method is used to perform back testing using the CPU by applying 3 filters:
        date, max and min price, and get the firs coincidence.

        Args:
            parameters (pl.DataFrame): parameters to make the review
            df_1m (pl.DataFrame): the data base
        Return:
            pl.DataFrame: the result of the back testing
        """
        # declare variables
        entry_time = parameters["entry_time"]
        max_v = parameters["min"]
        min_v = parameters["max"]
        date_max = parameters["Date"]
        # list for storing the data frames of one row
        data_frames = []
        # Select only the necessary columns
        df_1m_limits = df_1m.select(["Date_max", "High", "Low"])
        # Date max is rewrite to avoid repetitions
        df_1m_limits = df1_1m.select(["Date_max])
        # loop to return the closing date of the trade
        for entry_time_i, max_i, min_i, date_max_i in zip(
            entry_time, max_v, min_v, date_max
        ):

            # print(f"Date Range: {entry_time_i} TO {date_max_i}\nPrice Range: {min_i} TO {max_i}")

            # First, filter by time. Filter the database
            df_filtered_by_time = df_1m_limits.filter(
                (pl.col("Date") >= entry_time_i) & (pl.col("Date") <= date_max_i)
            )
            # Second, filter by high and low. Filter the database
            df = df_filtered_by_time.filter(
                (pl.col("High") >= max_i) | (pl.col("Low") <= min_i)
            )
            # Third, get only the first value that fulfils the condition
            df = df.head(1)
            # if null, then close the order within the maximum time limit.
            if df.is_empty():
                df = df_filtered_by_time.select(pl.all().last())

            # Fourth, stores the data frame in a list
            data_frames.append(df)
        # Join data frames
        df_concat = pl.concat(data_frames)
        return df_concat

    def _concat_parameters_with_back(
        self, back_testing: pl.DataFrame, parameters: pl.DataFrame
    ) -> pl.DataFrame:
        """_summary_
        Concatenate the parameters with the back testing results
        Args:
            back_testing (pl.DataFrame): The dataframe with the back testing performed
            parameters (pl.DataFrame): The input parameters of the strategy

        Returns:
            pl.DataFrame: A dataframe resulting from a concatenation
        """
        # rename column 'Date' to 'exit_time'.
        back_testing = back_testing.rename(
            {
                "Date": "exit_time",
            }
        )
        # Concatenate both data frames, 'back_testing' and 'parameters'.
        back_testing = pl.concat([parameters, back_testing], how="horizontal")
        return back_testing

    def _add_close_price(self, df: pl.DataFrame) -> pl.DataFrame:
        """_summary_
        Add close price to the data frame.
        Args:
            df (pl.DataFrame): Dataframe with upper and lower limits columns

        Returns:
            pl.DataFrame: A dataframe with a new column 'close_price'.
        """
        # getting close price column
        df = df.with_columns(
            pl.when(pl.col("High") >= pl.col("max"))  # condition
            .then(pl.col("max"))  # max is tp if it's buy but sl if it's sell
            .otherwise(pl.col("min"))  # min is sl if it's buy but tp if it's sell
            .alias("close_price")  # name the column created
        )
        return df

    def _add_type_order(self, df: pl.DataFrame) -> pl.DataFrame:
        """_summary_
        Add type order to the data frame.
        Args:
            df (pl.DataFrame): Dataframe with tp and sl columns

        Returns:
            pl.DataFrame: A dataframe with a new column 'type'.
        """
        # getting type order price
        df = df.with_columns(
            pl.when(pl.col("tp") > pl.col("sl"))  # condition
            .then(pl.lit("buy"))  # if tp > sl, then buy
            .otherwise(pl.lit("sell"))  # otherwise sell
            .alias("max")  # replace the result in a specific column
        )
        df = df.rename(
            {
                "max": "type",  # rename the column created
            }
        )
        return df

    def _add_utility_price(self, df: pl.DataFrame) -> pl.DataFrame:
        """_summary_
        Add the profit of each operation
        Args:
            df (pl.DataFrame): Dataframe with type, tp, sl and close_price columns

        Returns:
            pl.DataFrame: A dataframe with a new column 'type'.
        """
        # getting utility price
        df = df.with_columns(
            pl.when(
                # Condition for buy orders (buy)
                (
                    (pl.col("type") == pl.lit("buy"))  # If the order type is 'buy'
                    & (
                        pl.col("close_price") > pl.col("Open")
                    )  # And 'close_price' is higher than 'Open'
                )
                # Condition for sell orders (sell)
                | (
                    (pl.col("type") == pl.lit("sell"))  # If the order type is 'sell'
                    & (
                        pl.col("close_price") < pl.col("Open")
                    )  # And 'close_price' is lower than 'Open'
                )
            )
            # If the first or second condition is True,
            # calculate the profit as the absolute difference
            # between 'Open' and the 'tp'.
            .then(abs(pl.col("Open") - pl.col("tp")))
            # If the first or second condition is not True,
            # calculate the profit as the absolute negative difference
            # between 'Open' and the 'sl'.
            .otherwise(-abs(pl.col("Open") - pl.col("sl"))).alias(
                "utility"
            )  # name the column created
        )
        return df

    def _add_cumulative_utility(self, df: pl.DataFrame) -> pl.DataFrame:
        """_summary_
        Add cumulative utility to the data frame.
        Args:
            df (pl.DataFrame): Dataframe with utility column

        Returns:
            pl.DataFrame: A dataframe with a new column 'cumulative_utility'.
        """
        # profit accumulation
        df = df.with_columns(
            pl.col("utility")
            .cum_sum()
            .alias("Cumulative_utility")  # calculate cumulative sum
        )
        return df

    def _drop_unnecessary_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """_summary_

        Args:
            df (pl.DataFrame): A DataFrame to be transformed

        Returns:
            pl.DataFrame: A DataFrame without unnecessary columns
        """
        df = df.drop(["High", "Low", "min", "Date_max"])
        return df

    def get_journal(
        self, parameters: pl.DataFrame, df_1m: pl.DataFrame
    ) -> pl.DataFrame:
        """_summary_
        This method is used to perform trading journal using the CPU.

        Args:
            parameters (pl.DataFrame): parameters to make the review
            df_1m (pl.DataFrame): the principal and most important data base
        Return:
            pl.DataFrame: The trading journal
        """
        # Add 'min' and 'max' columns
        parameters = self._get_max_min(parameters)
        # Do back testing and write it into a dataframe called 'back_test'
        back_test = self._run_back_test(parameters, df_1m)
        # Concat the 'back_test' dataframe with 'parameters' dataframe
        back_test_concat = self._concat_parameters_with_back(back_test, parameters)
        # Add close price column
        back_test_concat = self._add_close_price(back_test_concat)
        # Add type order column
        back_test_concat = self._add_type_order(back_test_concat)
        # Add utility price column
        back_test_concat = self._add_utility_price(back_test_concat)
        # Add cumulative utility column
        back_test_concat = self._add_cumulative_utility(back_test_concat)
        # Delete unnecessary columns
        trading_journal = self._drop_unnecessary_columns(back_test_concat)
        # Return the trading journal
        return trading_journal

    # Make Back testing using GPU (Faster than CPU)
    def back_tester_gpu(self, parameters, df_1m):
        """_summary_
        Parameters

        Args:
            parameters (pl.DataFrame): _description_
            df_1m (pl.DataFrame): _description_
        """
        # <==============================================================>#
        # <==================== GPU BACK TESTER =========================>#
        # <==============================================================>#


# This class is in charge of data visualization.
class DataVisualizer:
    """
    Resume:
    This class is responsible for the visualization of the data after processing.
    """

    def __init__(self):
        self.logo_ascii()
        init()  # colorama

    # this is just for decorative purpose
    def logo_ascii(self) -> None:
        """
        Resume:
        Display the logo of the application
        """
        # Gold / Dollar
        print(Fore.YELLOW)
        print("<================================================>")
        print("██╗  ██╗ █████╗ ██╗   ██╗██╗   ██╗███████╗██████╗")
        print("╚██╗██╔╝██╔══██╗██║   ██║██║   ██║██╔════╝██╔══██╗")
        print(" ╚███╔╝ ███████║██║   ██║██║   ██║███████╗██║  ██║")
        print(" ██╔██╗ ██╔══██║██║   ██║██║   ██║╚════██║██║  ██║")
        print("██╔╝ ██╗██║  ██║╚██████╔╝╚██████╔╝███████║██████╔╝")
        print("╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚═════╝")
        print("<======== Created By Joel Pasapera Pinto ========>")
        print("<===== Follow me: github.com/JoelPasapera ======>")
        print("<================================================>")
        print(Style.RESET_ALL)

    def write_csv(
        self, df: pl.DataFrame, file_name: str = "back_test", show: bool = False
    ) -> None:
        """_summary_

        Args:
            df (pl.DataFrame): Dataframe to be written into CSV file
            file_name (str, optional): Name of the output CSV file. Defaults to "back_test".
            show (bool, optional): Show results if you want. Defaults to False.
        """
        # Write a DataFrame to a CSV file
        # < ============================================================>#
        # <==================== SAVE IN CVS FILE =======================>#
        # < ============================================================>#
        # In order to avoid wrong formats in 'Date' column, time is converted to text.
        df = df.with_columns(
            pl.col(["entry_time", "exit_time"]).dt.to_string("%Y-%m-%d %H:%M:%S")
        )
        # generate the path
        csv_path = f"{file_name}.csv"
        # create a csv file by writing the data frame
        df.write_csv(pathlib.Path(csv_path))
        if show is True:
            print(f"Data saved in {csv_path}")
            print(df)

    def save_graph_as_png(
        self,
        df: pl.DataFrame,
        file_name: str = "chart",
        x_title: str = "exit_time",
        y_title: str = "Cumulative_utility",
    ) -> None:
        """_summary_

        Args:
            df (pl.DataFrame): Dataframe to be graph and save as png file
            file_name (str, optional): Name of the output png file. Defaults to "chart".
            title (str, optional): Title of the graph. Defaults to "Graph".
            x_title (str, optional): Dataframe's name column that will be x axis.
            Defaults to "exit_time".
            y_title (str, optional): Dataframe's name column that will be y axis.
            Defaults to "Cumulative_utility".
        """
        # < ============================================================>#
        # < =============== SAVE UTILITY IN PNG FILE ===================>#
        # < ============================================================>#
        # create a new figure
        # create figure
        fig = go.Figure(
            go.Scatter(
                x=df[x_title],
                y=df[y_title],
                mode="lines",
                name=file_name,
            )
        )

        # Update the designed of the graph (optional)
        fig.update_layout(
            title=file_name,
            xaxis_title=x_title,
            yaxis_title=y_title,
            plot_bgcolor="white",
        )
        # save the graph with png extension
        image_path = f"{file_name}.png"
        pio.write_image(fig, image_path)

    def generate_pdf_report(self):
        """_summary_
        Generate analysis reports in pdf format (summary)
        """
        print("Implementation coming soon...")


# evaluates technical aspects of a trading strategy,
class StrategyEvaluator:
    """
    Resume:
    Evaluates technical aspects of a trading strategy.
    """

    # such as performing Monte Carlo tests and obtaining metrics such as profit factor and win rate
    def __init__(self, df):
        self.df = df


# Give appropriate advice according to the trader's profile
class InvestorProfileRecommender:
    """
    Resume:
    This class will make recommendations on how to set the parameters of
    trading strategies according to the investor's profile.
    Whether low, medium or high risk
    after the strategy demonstrate that is profitable
    """

    def __init__(self, df):
        self.df = df


# make back testing with million of different parameters until found the right strategy
class IntensiveBackTesting:
    """
    Intensive back testing of a trading strategy with a large number of parameters.
    This will be similar to the main function in that it will execute pretty much
    everything except loading and scanning csv files, which it will do in a loop
    iterating through different combinations of possible parameters.
    """

    def __init__(self, df, strategy, risk_management, data_visualizer, strategy_eval):
        self.df = df
        self.strategy = strategy
        self.risk_management = risk_management
        self.data_visualizer = data_visualizer
        self.strategy_eval = strategy_eval


# This is the main function
def main():
    """
    Main function to run data processing, analysis, and visualization for the financial asset.
    This function orchestrates the workflow from data loading to result visualization.
    """
    # the intention of this is to run all scripts in the same location as this script.
    setup_file_paths()
    # < ============================================================>#
    # < ============== DATA CLEANING AND PREPARATION ===============>#
    # < ============================================================>#
    # Create an objet 'xauusd' which represents gold data
    xauusd = CSVProcessor(csv_name="XAUUSD")
    # Create data frames (pl.Dataframe) for xauusd
    df_1m = xauusd.scan_csv(time_frame="1m")  # 1-minute (1m) candlestick
    ### df_5m = xauusd.scan_csv(time_frame="5m")  # 5-minute (5m) candlestick
    # < ============================================================>#
    # < ====================== DATA ANALYSIS =======================>#
    # < ============================================================>#
    # Get parameters of each strategy
    get_parameters = StrategyFactory()
    # Strategy 1
    parameters_1 = get_parameters.strategy_1(df_1m)
    # Strategy 2
    ### parameters_2 = get_parameters.strategy_2(df_5m)
    # Materialize before make back testing (THIS IS MANDATORY)
    df_1m = df_1m.collect()
    # make the back testing
    review = BackTesting()
    back_test = review.get_journal(parameters=parameters_1, df_1m=df_1m)
    # < ============================================================>#
    # < =================== DATA VISUALIZATION =====================>#
    # < ============================================================>#
    # png, csv and pdf files are generated for the presentation of the data.
    view = DataVisualizer()
    # generate csv file
    view.write_csv(df=back_test, file_name="XAUUSD_back_testing")
    # generate png file
    view.save_graph_as_png(df=back_test, file_name="XAUUSD_utility")
    # generate pdf file
    view.generate_pdf_report()


if __name__ == "__main__":
    main()
