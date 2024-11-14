# ExpertAdvisory
ExpertAdvisory is a backtesting tool intended for traders who want to evaluate scalping and intraday trading strategies using historical data quickly and efficiently. This project utilizes Python and advanced libraries such as Polars and PyTorch to achieve execution times in just a few seconds.
## Features
### Fast Processing:
- Performs 14-year backtesting in 30 seconds using polars.
### Multi-Strategy Support: 
- Allows you to define and test a variety of trading strategies.
### Visual Analysis: 
- Includes tools to visualise backtesting results using Plotly and Polars.
### Efficient Data Management:
- Uses Polars for efficient handling of large volumes of CSV data.
## Installation
Instructions on how to install the necessary dependencies. Do this: pip install -r requirements.txt
## Usage
Examples on how to use the script for backtesting in the following video https://www.youtube.com/watch?v=6wFukUeQCcg&ab_channel=JoelPasapera 
## Limitations
This software is specifically optimised to run and test Scalping (with a minimum duration of one minute) and Intra Day (with a maximum trading time of one day) strategies.
It is important to note that the software is not designed for:
### High Frequency Trading (HFT):
- It is not suitable for multiple trades at extremely short intervals.
### Swing Trading:
- It does not support strategies that involve holding open positions for several days or weeks.
## Contributions
Contributions are welcome. If you wish to contribute, please open an issue or send a pull request.
## License
This project is licensed under Aapache 2.0.
