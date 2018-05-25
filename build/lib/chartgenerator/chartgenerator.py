import plotly.offline as py
import plotly.plotly as pp
import plotly.graph_objs as go
import plotly.tools as pt

import configparser
import datetime
import logging
import os
from pprint import pprint
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import peakutils

import boto3

#from flask import Flask, Markup, redirect, render_template, request, url_for
#app = Flask(__name__)

#logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

config_path = 'config/config.ini'


class ChartGenerator:
    candle_directory = 'json/candles/'
    html_directory = 'html/'
    image_directory = 'png/'

    html_static_path = 'chart_current.html'

    s3_bucket_name = 'teslabot'
    s3_chart_name = 'chart.html'

    candle_types = ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']

    html_out = ''

    def __init__(self, trim_length_hourly, trim_length_daily, pretrim=0,
                 output_html=False, render_html=False, output_png=False,
                 json_file=None,
                 candle_dir=None,
                 image_dir=None,
                 keep_json_files=False,
                 random_pair_samples=None):

        self.candles = {}

        self.trim_length_hourly = trim_length_hourly
        self.trim_length_daily = trim_length_daily

        self.pretrim = pretrim

        self.output_html = output_html
        self.render_html = render_html
        self.output_png = output_png

        self.random_pair_samples = random_pair_samples

        if self.render_html == True or self.output_png == True:
            self.s3 = boto3.resource('s3')

        config = configparser.ConfigParser()
        config.read(config_path)

        plotly_user = config['plotly']['user']
        plotly_key = config['plotly']['key']

        pt.set_credentials_file(username=plotly_user, api_key=plotly_key)

        """
        if json_files != None:
            if not isinstance(json_files, list):
                logger.error('JSON files must be provided as a list. Exiting.')

                sys.exit(1)

            else:
                self.json_files = json_files
        """

        if candle_dir != None:
            if candle_dir[-1] != '/':
                candle_dir += '/'

            ChartGenerator.candle_directory = candle_dir

            #ChartGenerator.candle_complete_directory = ChartGenerator.candle_directory + 'complete/'

        if json_file == None and not os.path.exists(ChartGenerator.candle_directory):
            logger.error('Candle directory "' + ChartGenerator.candle_directory + '" not found. Exiting.')

            sys.exit(1)

        else:
            if json_file != None:
                self.json_file = json_file

            #if not os.path.exists(ChartGenerator.candle_complete_directory):
                #logger.info('Creating directory to store json files after successful analysis.')

                #os.mkdir(ChartGenerator.candle_complete_directory)

            if self.output_html == True and not os.path.exists(ChartGenerator.html_directory):
                logger.info('Creating directory to output chart HTML files.')

                os.mkdir(ChartGenerator.html_directory)

            if self.output_png == True and not os.path.exists(ChartGenerator.image_directory):
                logger.info('Creating directory to output chart PNG files.')

                os.mkdir(ChartGenerator.image_directory)

            self.keep_json_files = keep_json_files

            if self.keep_json_files == False:
                candle_archive_dir_base = candle_dir + 'archive/'

                if not os.path.exists(candle_archive_dir_base):
                    os.mkdir(candle_archive_dir_base)

                self.candle_archive_directory = candle_archive_dir_base + datetime.datetime.now().strftime('%m%d%Y-%H%M%S') + '/'

                os.mkdir(self.candle_archive_directory)

        self.html_div_output = ''

        self.include_js = True

        self.chart_div_collection = []

        #self.create_charts()


    def create_charts(self):
        def parse_file_path(candle_file):
            #logger.debug('candle_file: ' + candle_file)

            if self.json_file == None:
                candle_path = ChartGenerator.candle_directory + candle_file

            else:
                candle_path = candle_file
            #logger.debug('candle_path: ' + candle_path)

            market_info = candle_file.split('_')[1]

            exchange = market_info.split('-')[0]
            #logger.debug('exchange: ' + exchange)

            market = market_info.split('-')[1].upper()
            #logger.debug('market: ' + market)

            interval = candle_file.split('_')[2].split('.')[0].capitalize()
            #logger.debug('interval: ' + interval)

            img_path = ChartGenerator.image_directory + candle_path.split('/')[-1].rstrip('.json') + '.png'
            #logger.debug('img_path: ' + img_path)

            html_path = img_path.rstrip('.png') + '.html'
            #logger.debug('html_path: ' + html_path)

            return (candle_path, exchange, market, interval, img_path, html_path)


        def format_market_name(market):
            #logger.debug('market: ' + market)

            market_unformatted = market.lower()
            #logger.debug('market_unformatted: ' + market_unformatted)

            trade_product = ''
            quote_product = ''

            if market_unformatted[-1] == 'c':
                trade_product = market_unformatted[:-3].upper()
                quote_product = 'BTC'

            elif market_unformatted[-1] == 'h':
                trade_product = market_unformatted[:-3].upper()
                quote_product = 'ETH'

            elif market_unformatted[-1] == 'r':
                trade_product = market_unformatted[:-3].upper()
                quote_product = 'XMR'

            elif market_unformatted[-1] == 't':
                trade_product = market_unformatted[:-4].upper()
                quote_product = 'USDT'

            #logger.debug('trade_product: ' + trade_product)
            #logger.debug('quote_product: ' + quote_product)

            if trade_product != '' and quote_product != '':
                market_name = trade_product + '/' + quote_product

            else:
                logger.warning('Could not format market name.')

                market_name = market_unformatted.upper()

            logger.debug('market_name: ' + market_name)

            return market_name


        chart_return = []

        candles_json_selected = []

        if self.json_file == None:
            candles_dir_contents = os.listdir(ChartGenerator.candle_directory)

            candles_json_files = []
            for file in candles_dir_contents:
                if '.json' in file:
                    candles_json_files.append(file)

            if self.random_pair_samples != None:
                logger.debug('Using random samples.')

                if isinstance(self.random_pair_samples, int):
                    while (True):
                        rand = random.randint(0, (len(candles_json_files) - 1))
                        #logger.debug(rand)

                        if rand % 2 == 0:
                            daily = candles_json_files[rand]
                            hourly = candles_json_files[rand + 1]

                            candles_json_selected.append(daily)
                            candles_json_selected.append(hourly)

                            #logger.debug('Length Selected: ' + str(len(candles_json_selected)))

                            candles_json_files.remove(daily)
                            candles_json_files.remove(hourly)

                            #logger.debug('Length Files: ' + str(len(candles_json_files)))

                        if len(candles_json_selected) == (self.random_pair_samples * 2):
                            break

                else:
                    logger.error('Random pair count must be provided as an integer. Exiting.')

                    sys.exit(1)

            else:
                logger.debug('Using all files in candle directory.')

                candles_json_selected = candles_json_files

        else:
            logger.debug('Using JSON file list.')

            candles_json_selected.append(self.json_file)

        candles_json_selected.sort()

        for candles_json in candles_json_selected:
            print(candles_json)

            chart_current = {}

            file_data = parse_file_path(candles_json)
            # Returns: (candle_path, exchange, market, interval, img_path)

            exchange_name = file_data[1]
            logger.debug('exchange_name: ' + exchange_name)

            market_name_formatted = format_market_name(file_data[2])
            logger.debug('market_name_formatted: ' + market_name_formatted)

            market_title = exchange_name + '-' + market_name_formatted
            logger.debug('market_title: ' + market_title)

            market_interval = file_data[3]
            logger.debug('market_interval: ' + market_interval)

            image_path = file_data[4]
            logger.debug('image_path: ' + image_path)

            html_path = file_data[5]
            logger.debug('html_path: ' + html_path)

            chart_title = market_title + ' (' + market_interval + ')'
            logger.info(chart_title)

            chart_current['json'] = file_data[0]
            chart_current['exchange'] = file_data[1]
            chart_current['market'] = file_data[2]
            chart_current['interval'] = file_data[3]
            chart_current['title'] = chart_title
            chart_current['results'] = {'resistance': None, 'resistance_info': '',
                                        'support': None, 'support_info': ''}
            chart_current['image'] = {'path': image_path, 'url': None}
            chart_current['html'] = {'path': html_path, 'url': None}
            chart_current['s3_ready'] = False

            if market_interval.lower() == 'hourly':
                interval_hours = 1

            else:
                interval_hours = 24

            candles_pandas = pd.read_json(file_data[0])  # Read JSON files into Pandas

            for data_type in ChartGenerator.candle_types:
                logger.debug('data_type: ' + data_type)
                logger.debug('exchange_name.lower(): ' + exchange_name.lower())

                if data_type == 'open_time':
                    if exchange_name.lower() == 'bittrex':
                        # Bittrex uses millisecond timestamps
                        # 1 hour = 3,600,000 ms
                        # 1 day = 86,400,000 ms
                        # Binance uses microsecond timestamps
                        # 1 hour = 3,600,000,000 us
                        # 1 day = 86,400,000,000 us

                        close_times = candles_pandas['close_time'].tolist()

                        self.candles['open_time'] = []

                        for x in range(0, len(close_times)):
                            self.candles['open_time'].append(close_times[x] - pd.Timedelta(hours=interval_hours))

                    else:
                        self.candles['open_time'] = candles_pandas['open_time'].tolist()

                elif data_type == 'close_time':
                    if exchange_name.lower() == 'bittrex':
                        close_times = candles_pandas['close_time'].tolist()

                        self.candles['close_time'] = close_times

                        self.candles['open_time'] = []

                        for x in range(0, len(close_times)):
                            self.candles['open_time'].append(close_times[x] - pd.Timedelta(hours=interval_hours))

                    else:
                        close_times = candles_pandas['close_time'].tolist()

                        self.candles['close_time'] = []

                        for x in range(0, len(close_times)):
                            self.candles['close_time'].append(close_times[x] + pd.Timedelta(milliseconds=1))

                else:
                    if exchange_name.lower() == 'bittrex':
                        self.candles[data_type] = candles_pandas[data_type].tolist()

                    else:
                        self.candles[data_type] = candles_pandas[data_type].tolist()

            for data_type in ChartGenerator.candle_types:
                #self.candles[data_type] = self.candles[data_type][::-1][self.pretrim:]
                self.candles[data_type] = self.candles[data_type][self.pretrim:][::-1]

            ## Get and format all required data for analysis ##

            if market_interval.lower() == 'hourly':
                trim_length = int(self.trim_length_hourly)# * -1)

            else:
                trim_length = int(self.trim_length_daily)# * -1)

            #time_x = pd.to_datetime(self.candles['open_time'])[:trim_length]
            time_x = pd.to_datetime(self.candles['close_time'])[:trim_length]

            open_y = np.array(self.candles['open'])[:trim_length]
            open_x = np.array([j for j in range(len(open_y))])

            open_last = open_y[-1]
            logger.debug('open_last: ' + str(open_last))

            open_max = max(open_y)
            logger.debug('open_max: ' + str(open_max))

            open_min = min(open_y)
            logger.debug('open_min: ' + str(open_min))

            high_y = np.array(self.candles['high'])[:trim_length]
            high_x = np.array([j for j in range(len(high_y))])

            high_first = high_y[0]
            logger.debug('high_first: ' + str(high_first))

            high_last = high_y[-1]
            logger.debug('high_last: ' + str(high_last))

            high_max = max(high_y)
            logger.debug('high_max: ' + str(high_max))

            high_min = min(high_y)
            logger.debug('high_min: ' + str(high_min))

            low_y = np.array(self.candles['low'])[:trim_length]
            low_x = np.array([j for j in range(len(low_y))])

            low_last = low_y[-1]
            logger.debug('low_last: ' + str(low_last))

            low_max = max(low_y)
            logger.debug('low_max: ' + str(low_max))

            low_min = min(low_y)
            logger.debug('low_min: ' + str(low_min))

            close_y = np.array(self.candles['close'])[:trim_length]
            close_x = np.array([j for j in range(len(close_y))])

            close_last = close_y[-1]
            logger.debug('close_last: ' + str(close_last))

            close_max = max(close_y)
            logger.debug('close_max: ' + str(close_max))

            close_min = min(close_y)
            logger.debug('close_min: ' + str(close_min))

            volume_y = np.array(self.candles['volume'])[:trim_length]
            volume_x = np.array([j for j in range(len(volume_y))])

            ## Calculate Peaks ##

            peaks_thres = round((close_last - low_min) / (high_max - low_min), 3)
            logger.debug('peaks_thres: ' + str(peaks_thres))

            peaks_min_dist = 0.1

            while (True):
                peaks_x = peakutils.indexes(high_y, thres=peaks_thres, min_dist=peaks_min_dist)

                peak_count = len(peaks_x)

                if peak_count > 50:
                    peaks_min_dist += 0.0025

                elif peak_count > 5:
                    peaks_thres += 0.0025

                else:
                    break

            logger.debug('peaks_thres: ' + str(peaks_thres))
            logger.debug('peaks_min_dist: ' + str(peaks_min_dist))

            peaks_y = [high_y[j] for j in peaks_x]

            ## Calculate Troughs ##

            troughs_thres = round((close_last - low_min) / (high_max - low_min), 3)
            logger.debug('troughs_thres: ' + str(troughs_thres))

            troughs_min_dist = 0.1

            low_y_flipped = np.multiply(low_y, -1)

            while (True):
                troughs_x = peakutils.indexes(low_y_flipped, thres=troughs_thres, min_dist=troughs_min_dist)

                trough_count = len(troughs_x)

                if trough_count > 50:
                    troughs_min_dist += 0.0025

                elif trough_count > 5:
                    troughs_thres += 0.0025

                else:
                    break

            logger.debug('troughs_thres: ' + str(troughs_thres))
            logger.debug('troughs_min_dist: ' + str(troughs_min_dist))

            troughs_y = [low_y[j] for j in troughs_x]

            ## Calculate Resistance Level ##

            if len(peaks_y) > 1:
                resistance = (peaks_y[0] + peaks_y[1]) / 2

            elif len(peaks_y) == 1:
                logger.warning('Only one peak detected. Using as resistance level.')  # AVERAGE W/ MAX VAL?

                chart_current['results']['resistance_info'] = 'Only one peak detected. Using as resistance level.'

                resistance = peaks_y[0]

            else:
                logger.error('Y-Peaks length = ' + str(len(peaks_y)))

                logger.warning('Setting resistance to maximum high price.')

                chart_current['results']['resistance_info'] = 'No peaks detected. Using maximum price as resistance.'

                resistance = max(high_y)

            if resistance < close_last:
                logger.info('Price is trending and resistance lower than current price.\n' +
                            'Unable to calculate true resistance. Using current price as resistance.')

                chart_current['results']['resistance_info'] += ' (Adjusted to current close price)'

                resistance = close_last

            chart_current['results']['resistance'] = resistance

            ## Calculate Support Level ##

            if len(troughs_y) > 1:
                support = (troughs_y[0] + troughs_y[1]) / 2

            elif len(troughs_y) == 1:
                logger.warning('Only one trough detected. Using as support level.')  # AVERAGE W/ MIN VAL?

                chart_current['results']['support_info'] = 'Only one trough detected. Using as support level.'

                support = troughs_y[0]

            else:
                logger.error('Y-Troughs length = ' + str(len(troughs_y)))

                logger.warning('Setting support to minimum low price.')

                chart_current['results']['support_info'] = 'No troughs detected. Using minimum price as support.'

                support = min(low_y)

            if support > close_last:
                logger.info('Price is trending and support higher than current price.\n' +
                            'Unable to calculate true support. Using current price as support.')

                chart_current['results']['support_info'] += ' (Adjusted to current close price)'

                support = close_last

            chart_current['results']['support'] = support

            logger.info('Support: ' + "{:.8f}".format(support))
            logger.info('Resistance: ' + "{:.8f}".format(resistance))

            ## Plot OHLC Candles ##

            ohlc_traces = []

            logger.debug('open_y[0]: ' + "{:.8f}".format(open_y[0]))
            logger.debug('open_y[-1]: ' + "{:.8f}".format(open_y[-1]))
            logger.debug('high_y[0]: ' + "{:.8f}".format(high_y[0]))
            logger.debug('high_y[-1]: ' + "{:.8f}".format(high_y[-1]))
            logger.debug('low_y[0]: ' + "{:.8f}".format(low_y[0]))
            logger.debug('low_y[-1]: ' + "{:.8f}".format(low_y[-1]))
            logger.debug('close_y[0]: ' + "{:.8f}".format(close_y[0]))
            logger.debug('close_y[-1]: ' + "{:.8f}".format(close_y[-1]))

            ohlc_traces.append(go.Candlestick(
                x = time_x,
                open = open_y,
                high = high_y,
                low = low_y,
                close = close_y,
                increasing=dict(line=dict(color='rgba(0,255,0,100)'), fillcolor='rgba(0,255,0,100)'),
                decreasing=dict(line=dict(color='rgba(255,0,0,100)'), fillcolor='rgba(255,0,0,100)'),
                yaxis = 'y2',
                name='OHLC',
                showlegend=False
            ))

            ohlc_traces.append(go.Scatter(
                #x = [time_x[0], time_x[-1]],
                x = [(time_x[0] + abs(time_x[0] - time_x[1])), (time_x[-1] - abs(time_x[-1] - time_x[-2]))],
                y = [resistance, resistance],
                mode = 'lines',
                line = {'color': 'rgba(255,0,0,100)', 'width': 4},
                yaxis = 'y2',
                name = 'Resistance'
            ))

            ohlc_traces.append(go.Scatter(
                #x = [time_x[0], time_x[-1]],
                x = [(time_x[0] + (abs(time_x[1] - time_x[0]))), (time_x[-1] - (abs(time_x[-1] - time_x[-2])))],
                y = [support, support],
                mode = 'lines',
                line = {'color': 'rgba(0,255,0,100)', 'width': 4},
                yaxis = 'y2',
                name = 'Support'
            ))

            fig = dict(data=ohlc_traces)

            fig['layout'] = dict()
            fig['layout']['title'] = chart_title
            fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,100)'
            fig['layout']['xaxis'] = dict(title='Date/Time',
                                          rangeselector=dict(visible=True),
                                          gridcolor='rgba(100,100,100,100)',
                                          gridwidth=1,
                                          showgrid=True,
                                          linecolor='rgba(255,255,255,100)',
                                          #linewidth=0.5,
                                          showline=True)
            fig['layout']['yaxis'] = dict(domain=[0, 0.1], showticklabels=False)
            fig['layout']['yaxis2'] = dict(domain=[0.1, 1],
                                           gridcolor='rgba(100,100,100,100)',
                                           gridwidth=1,
                                           showgrid=True,
                                           linecolor='rgba(255,255,255,100)',
                                           #linewidth=0.5,
                                           showline=True)
            fig['layout']['legend'] = dict(orientation='h',
                                           x=1, xanchor='right',
                                           y=1, yanchor='bottom',
                                           traceorder='reversed')
            fig['layout']['margin'] = dict(t=60, b=60, r=40, l=60)

            volume_colors = []

            for i in range(len(close_y)):
                if i != 0:
                    if close_y[i] > close_y[i - 1]:
                        volume_colors.append('rgba(0,255,0,100)')

                    else:
                        volume_colors.append('rgba(255,0,0,100)')

                else:
                    volume_colors.append('rgba(255,0,0,100)')

            fig['data'].append(dict(x=time_x,
                                    y=volume_y,
                                    marker=dict(color=volume_colors),
                                    type='bar',
                                    yaxis='y',
                                    name='Volume',
                                    showlegend=False
                                   ))

            if self.output_png == True:
                pp.image.save_as(fig, filename=image_path)

                #chart_current['image'] = {}

                #chart_current['image']['path'] = image_path

                s3_image_path = 'charts/' + image_path.split('/')[-1]
                logger.debug('s3_image_path: ' + s3_image_path)

                s3_image_url = 'http://teslabot.s3-website-us-east-1.amazonaws.com/charts/' + image_path.split('/')[-1]
                logger.debug('s3_image_url: ' + s3_image_url)

                self.s3.meta.client.upload_file(image_path,
                                                ChartGenerator.s3_bucket_name,
                                                s3_image_path,
                                                #ExtraArgs={'ContentType': 'text/html'})
                                                ExtraArgs={'ContentType': 'image/png'})

                chart_current['image']['url'] = s3_image_url

            range_buttons = []

            range_buttons.append(dict(count=1, label='reset', step='all'))

            if market_interval.lower() == 'hourly':
                range_buttons.append(dict(count=1, label='1 week', step='week', stepmode='backward'))
                range_buttons.append(dict(count=3, label='3 day', step='day', stepmode='backward'))
                range_buttons.append(dict(count=1, label='1 day', step='day', stepmode='backward'))

            if market_interval.lower() == 'daily':
                range_buttons.append(dict(count=1, label='1 year', step='year', stepmode='backward'))
                range_buttons.append(dict(count=3, label='3 month', step='month', stepmode='backward'))
                range_buttons.append(dict(count=1, label='1 month', step='month', stepmode='backward'))

            range_buttons.append(dict(step='all'))

            rangeselector=dict(
                visible = True,
                #x = 0, y = 0.9,
                bgcolor = 'rgba(150, 200, 250, 0.4)',
                font = dict(size=13),
                buttons=list(range_buttons)
            )

            fig['layout']['xaxis']['rangeselector'] = rangeselector

            #if display_plots == True:
                #py.iplot(fig, show_link=False)

            chart_div = py.plot(fig, include_plotlyjs=self.include_js, output_type='div', show_link=False)

            self.chart_div_collection.append(chart_div)

            if self.render_html == True:
                logger.debug('Writing chart HTML to file.')

                with open(html_path, 'w') as file:
                    file.write(chart_div)

                logger.info('Uploading chart to S3 bucket.')

                s3_html_path = 'charts/' + html_path.split('/')[-1]
                logger.debug('s3_html_path: ' + s3_html_path)

                s3_html_url = 'http://teslabot.s3-website-us-east-1.amazonaws.com/charts/' + html_path.split('/')[-1]
                logger.debug('s3_html_url: ' + s3_html_url)

                self.s3.meta.client.upload_file(html_path,
                                                ChartGenerator.s3_bucket_name,
                                                s3_html_path,
                                                ExtraArgs={'ContentType': 'text/html'})

                chart_current['html']['url'] = s3_html_url

            #if self.include_js == True:
                #self.include_js = False

            if self.keep_json_files == False:
                archive_path = self.candle_archive_directory + file_data[0].split('/')[-1]
                logger.debug('archive_path: ' + archive_path)

                shutil.move(file_data[0], archive_path)  # Move JSON file to archive directory after analysis complete

            chart_return.append(chart_current)

        self.html_div_output = ''

        for chart_div in self.chart_div_collection:
            self.html_div_output += '<br>' + chart_div + '<br>'

        if self.output_html == True:# or self.render_html == True:
            logger.info('Dumping charts to HTML file.')

            chart_html_path = ChartGenerator.html_directory + 'charts_' + datetime.datetime.now().strftime('%m%d%Y-%H%M%S') + '.html'

            with open(chart_html_path, 'w') as file:
                file.write(self.html_div_output)

            """
            if self.render_html == True:
                logger.debug('Writing HTML to static path.')

                with open(ChartGenerator.html_static_path, 'w') as file:
                    file.write(self.html_div_output)

                logger.info('Uploading chart to S3 bucket.')

                self.s3.meta.client.upload_file(ChartGenerator.html_static_path,
                                                ChartGenerator.s3_bucket_name,
                                                ChartGenerator.s3_chart_name,
                                                ExtraArgs={'ContentType': 'text/html'})
            """

        #if self.render_html == True:
            #ChartGenerator.html_out = self.html_div_output

        #chart_return['count'] = chart_count

        pprint(chart_return)

        return chart_return


"""
@app.route('/chart', methods=['GET', 'POST'])
def chart():
    error = None

    if request.method == 'GET':
        return render_template('chart_template.html', div_placeholder=Markup(ChartGenerator.html_out))

    elif request.method == 'POST':
        return redirect(url_for('chart', error=error))
"""


if __name__ == '__main__':
    chart_generator = ChartGenerator(100, 100, pretrim=1, output_html=False, render_html=True, output_png=True, candle_dir='json/test/')#, keep_json_files=True)#, candle_dir='json/candles/')

    chart_results = chart_generator.create_charts()

    #logger.debug('chart_results: ' + str(chart_results))
    print('CHART RESULTS:')
    pprint(chart_results)

    """
    if chart_generator.render_html == True:
        logger.info('Flask enabled.')

        app.run()#debug=True)

    else:
        logger.info('Flask disabled.')
    """