#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr
import mybatis_mapper2sql
import mysql.connector as mysqld
import argparse
import os, logging, time
import pandas as pd 
import ipaddress

__title__ = 'perftest'
__version__ = '0.2.0-DEV'
__author__ = 'khkwon01'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024'

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file",
        default="perftest.xml",
        nargs='?',
        type=str,
        help="file which was included test sql")
    parser.add_argument("--debug",
        default=False,
        action='store_true',
        help="Debug log enable.")    
    parser.add_argument("-h", "--help",
        default=False,
        action='store_true',
        help="show this help message and exit.")
    
    return parser

class MySQLdb:
    def __init__(self, ip, port, username, password):    
        self.db = None

        try:
            self.db = mysqld.connect(
                host=ip,
                port=port,
                user=username,
                passwd=password,
                connection_timeout=5,
                autocommit=True
            )
        except Exception as err:
            logging.exception(err)
            raise gr.Error(err)
        
    def execute_query(self, sql):
        cursor = self.db.cursor()
        init_time = time.time()
        cursor.execute(sql)
        after_time = time.time()
        res = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
        cursor.close()

        return res, (round(after_time - init_time,1))

    def disconnect(self):
        if self.db is not None:
            self.db.close()

class WebGui:
    def __init__(self, options):
        self.mapper, _ = mybatis_mapper2sql.create_mapper(xml=options.file)
        self.statement = mybatis_mapper2sql.get_child_statement(self.mapper, child_id='testsql')
        self.options = options

    def execute_testsql(self, service, ip, port, user, password, progress=gr.Progress()):
        response = None
        msg = None

        print(service, ip, port, user, password)
        progress(0, desc="Starting")

        if len(service) < 1 or len(user) < 1 or len(password) < 1:
            msg = f"The following value is mandatory for ip: {ip}, username: {user}, password: {password}"
            raise gr.Error(msg)
            logging.exception(msg)

        try:
            ipaddress.ip_address(ip)
        except ValueError:
            msg = f"This ip address is wrong format : {ip}"
            logging.exception(msg)
            raise gr.Error(message=msg)

        progress(0.3)
        mydb = MySQLdb(ip, port, user, password)
        data, exec_time = mydb.execute_query(self.statement)
        expl, exec_gar = mydb.execute_query('explain ' + self.statement)
        mydb.disconnect()
        
        resplot = gr.BarPlot(
            data,
            x="bookprice",
            y="num",
            title="How many book is per price",
            tooltip=["bookprice", "num"],
            y_lim=[100, 2000],
            width=500
        )

        progress(0.6) 

        if service == "mysql":
            response = {
                self.myresbox: msg if msg is not None else f"The exeuction of {service} is succeed!! (query execution time : {exec_time}, rowcount: {len(data)})", 
                self.myexpl: expl.to_markdown(),
                self.plotmysql: resplot              
            }
        else:
            response = {
                self.htresbox: msg if msg is not None else f"The exeuction of {service} is succeed!! (query execution time : {exec_time}, rowcount: {len(data)})", 
                self.htexpl: expl.to_markdown(),
                self.plotheatw: resplot               
            }

        progress(1) 

        return response
    
    def make_webui(self):
        self.iface = None

        with gr.Blocks() as self.iface:
            with gr.Row():
                gr.Markdown("## This is performance test between MySQL and HeatWave")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("- MySQL Info (OLTP)")
                    self.mysip = gr.Textbox(label="DB IP")
                    self.myport = gr.Number(label="DB Port", value=3306, interactive=True)
                    self.myuser = gr.Textbox(label="DB Username")
                    self.mypass = gr.Textbox(label="DB Password", type='password')
                with gr.Column(scale=1):
                    gr.Markdown("- HeatWave Info (OLAP)")
                    self.htip = gr.Textbox(label="DB IP")
                    self.htport = gr.Number(label="DB Port", value=3306, interactive=True)                        
                    self.htuser = gr.Textbox(label="DB Username")
                    self.htpass = gr.Textbox(label="DB Password", type='password')                    
            with gr.Row():
                with gr.Accordion("See the analytic query!", open=False):
                    gr.Code(self.statement)
            with gr.Row(equal_height=True):
                with gr.Column():
                    self.mybtn = gr.Button("execution")
                with gr.Column():
                    self.htbtn = gr.Button("execution")
            with gr.Row() as self.res:
                with gr.Column():
                    self.myresbox = gr.Textbox(label="mysql response")
                with gr.Column():
                    self.htresbox = gr.Textbox(label="heatwave response")
            with gr.Row():
                with gr.Column(scale=1):
                    self.myexpl = gr.Textbox(label="MySQL explain")
                with gr.Column(scale=1):
                    self.htexpl = gr.Textbox(label="heatwave explain")
            with gr.Row(equal_height=True) as self.output :
                with gr.Column(scale=1):
                    self.plotmysql = gr.BarPlot(label='MySQL')
                with gr.Column(scale=1):
                    self.plotheatw = gr.BarPlot(label='HeatWave')

            self.mybtn.click(self.execute_testsql, \
                            [gr.Textbox(value="mysql", visible=False), self.mysip, self.myport, self.myuser, self.mypass], \
                            [self.myresbox, self.myexpl, self.plotmysql])
            self.htbtn.click(self.execute_testsql, \
                            [gr.Textbox(value="heatwave", visible=False), self.htip, self.htport, self.htuser, self.htpass], \
                            [self.htresbox, self.htexpl, self.plotheatw])

        self.iface.launch(server_name="0.0.0.0", server_port=8000, show_error=True)

if __name__ == '__main__':
    parser = get_args_parser()
    options = parser.parse_args()
    if options.help:
        parser.print_help()
        parser.exit()

    if options.debug:
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        logging.basicConfig(
            format='%(asctime)s - (%(threadName)s) - %(message)s in %(funcName)s() at %(filename)s : %(lineno)s',
            level=logging.DEBUG,
            filename="logs/debug.log",
            filemode='w',
        )
        logging.debug(options)
    else:
        nl_hanlder = logging.NullHandler(logging.INFO)
        logging.basicConfig(handlers = [ nl_hanlder ])

    webgui = WebGui(options=options)
    webgui.make_webui()
