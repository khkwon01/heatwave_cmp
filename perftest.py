#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr
import mybatis_mapper2sql
import mysql.connector as mysqld
import argparse
import os, logging, time
import pandas as pd 
import ipaddress
import ocifs, json
from tabulate import tabulate
from googletrans import Translator


__title__ = 'perftest'
__version__ = '2.0.0-DEV'
__author__ = 'khkwon01'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025'


compatibility_heatwave = '9.3.0-cloud'
obj_model_list = [
                    "mistral-7b-instruct-v1",
                    "llama2-7b-v1",
                    "llama3-8b-instruct-v1",
#                    "cohere.command"
                 ]

obj_par_url = {
    "lkread"  : "https://objectstorage.ap-seoul-1.oraclecloud.com/p/uIbRJUZeFAcDFg8jGjEysP2_ojKyZ2-qPaC33CNtzFZfOFGYndokNrAnLCdPq-PV/n/apaccpt01/b/ml-lake-test/o/bank.csv",
    "lkwrite" : "https://objectstorage.ap-seoul-1.oraclecloud.com/p/YUmRUYyApkFUn_BuTvsKKGraOed8PjbUX27BmpQafNHWq1neJOmRGHmMVYuagQzp/n/apaccpt01/b/ml-lake-test/o/",
    "genai"   : "https://objectstorage.ap-seoul-1.oraclecloud.com/p/3uDqqopE400S-_JYTAcPWXkVjHlSSqT1z-aTOLQPRwUCOt6fWunMlnE8yTrlkrOg/n/apaccpt01/b/genai/o/mysql-heatwave-technical-brief.pdf"
}

similar_url="http://207.211.162.97:8501/"

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file",
        default="perftest.xml",
        nargs='?',
        type=str,
        help="File which was included test sql")
    parser.add_argument("--debug",
        default=False,
        action='store_true',
        help="Debug log enable.")    
    parser.add_argument("--server",
        default="",
        type=str,
        required=True,
        help="Heatwave server info (ex: ip,username,pass)")
    parser.add_argument("-o", "--oci",
        default="config/oci.conf",
        type=str,
        help="Oci config file for geting oci info")
    parser.add_argument("--namespace",
        default="apaccpt01",
        type=str,
        help="Namespace of oci object storage")
    parser.add_argument("--bucket",
        default="ml-lake-test",
        type=str,
        help="Bucket of oci object storage")
    parser.add_argument("-h", "--help",
        default=False,
        action='store_true',
        help="Show this help message and exit.")
    
    return parser

class MySQLdb:
    def __init__(self, ip, port, username, password, auto=True):    
        self.db = None

        try:
            self.db = mysqld.connect(
                host=ip,
                port=port,
                user=username,
                passwd=password,
                connection_timeout=10,
                autocommit=auto,
                time_zone='Asia/Seoul'
            )
        except mysqld.Error as err:
            logging.exception(err)
            #raise gr.Error(err)

    def is_connect(self):
        b_resp = True

        if self.db is None:
            b_resp = False

        return b_resp
        
    def execute_pd_query(self, sql):
        cursor = self.db.cursor()
        init_time = time.time()
        cursor.execute(sql)
        after_time = time.time()
        res = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
        cursor.close()

        return res, (round(after_time - init_time,1))

    def execute_ddl_query(self, sql):
        cursor = self.db.cursor()
        init_time = time.time()
        cursor.execute(sql)
        after_time = time.time()
        cursor.close()

        return (round(after_time - init_time,1))

    def execute_query(self, sql):
        #self.db.autocommit = False
        cursor = self.db.cursor(buffered=True)
        init_time = time.time()
        cursor.execute(sql, multi=True)
        cursor.close()
        after_time = time.time()

        return (round(after_time - init_time,1))

    def execute_affected_query(self, sql):

        cursor = self.db.cursor(buffered=True)
        init_time = time.time()
        cursor.execute(sql, multi=True)
        count = cursor.rowcount
        cursor.close()
        after_time = time.time()

        return (count, round(after_time - init_time,1))

    def execute_callproc(self, proc, args):
        o_out = ''
        o_resp = None

        cursor = self.db.cursor(buffered=True)
        init_time = time.time()
        o_resp = cursor.callproc(proc, args)
        after_time = time.time()
        for result in cursor.stored_results():
            for row in result.fetchall():
                o_out += ''.join(row)
                o_out += "\n"
        cursor.close()

        return o_out, o_resp, (round(after_time - init_time,1))

    def disconnect(self):
        if self.db is not None:
            self.db.close()

class WebGui:
    def __init__(self, options):
        self.options = options
        self.mapper, _ = mybatis_mapper2sql.create_mapper(xml=options.file)
        self.olapquery1 = mybatis_mapper2sql.get_child_statement(self.mapper, 
                result_type='raw', child_id='olapsql1')
        self.ragtable1 = mybatis_mapper2sql.get_child_statement(self.mapper, 
                child_id='ragtable1')
        self.hwcheck1 = mybatis_mapper2sql.get_child_statement(self.mapper,
                child_id='checkhw1')
        self.lkwrite1 = mybatis_mapper2sql.get_child_statement(self.mapper,
                child_id='lakewrite1')
        self.o_server = options.server.split(',')

        self.tranlat = Translator()

        try:
            self.o_fs = ocifs.OCIFileSystem(config=options.oci)
        except Exception as err:
            logging.exception(str(err))
            raise gr.Error(message=str(err))

    def check_hw_node(self):
        status = False

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        try:
            o_resp, exec_time = mydb.execute_pd_query(self.hwcheck1)
        except mysqld.Error as err:
            logging.exception(str(err))

        print(o_resp)
        if len(o_resp) > 0 and o_resp['status'].values[0] == 'AVAIL_RNSTATE':
            status = True

        mydb.disconnect()

        return status

    def execute_olap(self, service, ip, port, user, password, progress=gr.Progress()):
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
        data, exec_time = mydb.execute_pd_query(self.olapquery1)
        expl, exec_gar = mydb.execute_pd_query('explain format=tree ' + self.olapquery1)
        mydb.disconnect()
        
        resplot = gr.BarPlot(
            data,
            x="bookprice",
            y="num",
            title="How many book per price",
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

    def get_objcontents(self, objfile, delimiter):
        s_message = '> ### **Load bank csv data into heatwave**'

        if delimiter == "tab": delimiter='\t'

        if type(objfile) is not str:
            gr.Warning("You first select object file in radio component")
            return None, s_message

        o_bankpd = pd.read_csv("oci://" + objfile, sep=delimiter,
                      storage_options={"config": options.oci, 
                      "profile": "DEFAULT"})

        s_table = objfile.rsplit('/')[1].split('.')[0].replace("-","")
            
        return o_bankpd.head(10).to_markdown(), (s_message + "(" + objfile + ")"), s_table

    def load_objfiles(self):
        o_files = self.o_fs.ls(self.options.bucket + '@' + self.options.namespace, refresh=True)

        return [gr.Radio(o_files,label='csv files(obj)',
            info='files list in oci object storage'),None, None, None, None]

    def make_autoload_query(self, db, table, parurl, delimiter):
        s_autoload_query = None

        if len(db)==0 or len(table)==0 or len(parurl)==0 or len(delimiter)==0:
            raise gr.Error("You must input db and table, parul value")
            return None

        if delimiter == "tab": delimiter = "\\\\t" 

        # set db variable
        s_autoload_query = f'set @db_list = \'["{db}"]\';\n'

        # set table info
        s_autoload_query += ('set @ext_tables=\'[{"db_name":"' + db + '",')
        s_autoload_query += (' "tables":[{"table_name":"' + table +'",')
        s_autoload_query += (' "dialect":{"format":"csv","has_header":true,')
        s_autoload_query += (' "field_delimiter":"' + delimiter + '","record_delimiter":"\\\\n"},\n')
        s_autoload_query += (' "file": [{"par":"' + parurl + '"}] }] }]\';\n')

        # convert table info to json
        s_autoload_query += ("set @options=JSON_OBJECT('mode','normal','refresh_external_tables', TRUE,'external_tables',CAST(@ext_tables AS JSON));\n")
        s_autoload_query += ("call sys.heatwave_load(@db_list, @options);")

        return s_autoload_query

    def execute_autoload(self, db, table, parurl, delimiter):
        s_schema = "create database if not exists {};".format(db)

        if not self.check_hw_node():
            raise gr.Error("Firstly, you must setup the heatwave node")

        if len(db)==0 or len(table)==0 or len(parurl)==0 or len(delimiter)==0:
            raise gr.Error("You must input db and table, parul value")
            return None

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])
        exec_time = mydb.execute_ddl_query(s_schema)
        print(f"create time: {exec_time}")

        if delimiter == "tab": delimiter = "\t"

        options = {"mode": "normal", "external_tables": [{"tables": [{"file": [{"par": parurl}], "dialect": {"format": "csv", "has_header": True, "field_delimiter": delimiter, "record_delimiter": "\n"}, "table_name": table}], "db_name": db}]}

        o_out, _, exec_time = mydb.execute_callproc('sys.heatwave_load',(f'["{db}"]', json.dumps(options)))
        print(f"call proc time: {exec_time}")
        #time.sleep(5)
        
        chkquery = "select id, ts, stage, log from sys.heatwave_autopilot_report where comp = 'AUTO_LOAD' and stage = 'EXECUTE' and log->>'$.type' = 'table_summary';"

        o_ans, exec_time = mydb.execute_pd_query(chkquery)
        print(o_ans)

        mydb.disconnect()

        return o_out

    def make_automl_train(self, table):

        s_code = f"call sys.ML_TRAIN('{table}', 'y',\n     JSON_OBJECT('task', 'classification'), @bank_model);\n"

        s_code += "select @bank_mode;\n"

        return s_code

    def execute_automl_train(self, table):
        s_alogrithm = {"task": "classification"}
        s_model = f'{table}_model'
        s_delete_model = f"delete from ML_SCHEMA_{self.o_server[1]}.MODEL_CATALOG where model_handle = '{s_model}';"

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        # it don't matter when error happen
        try:
            exec_time = mydb.execute_query(s_delete_model)
        except mysqld.Error as err:
            logging.exception(err)

        try:
            o_out, o_resp, exec_time = mydb.execute_callproc('sys.ML_TRAIN',
                    (f'{table}', 'y', json.dumps(s_alogrithm), (s_model, 'CHAR')) )
        except mysqld.Error as err:
            logging.exception(err)
            raise gr.Error(err)

        mydb.disconnect()

        print('resp:', o_resp)

        return o_resp[3]

    def load_automl_model(self, model):
        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        try:

            o_out, o_resp, exec_time = mydb.execute_callproc('sys.ML_MODEL_LOAD',
                    (f'{model}', 'null'))

        except mysqld.Error as err:
            logging.exception(err)
            raise gr.Error(err)

        print('resp', o_resp)

        mydb.disconnect()

        return "Successful load : call sys.ML_MODEL_LOAD" + str(o_resp)

    def make_automl_predict(self, age, job, marital, edu, defa, bal, housing,
            loan, contact, day, month, dura, camp, pdays, prev, outcome, model):

        if len(model) < 1: model='AutoML.bankfull_model'

        s_predit_sql =  f'SELECT JSON_PRETTY(sys.ML_PREDICT_ROW(\n'
        s_predit_sql += f'    JSON_OBJECT("age", "{age}", "job", "{job}",\n'
        s_predit_sql += f'            "marital", "{marital}", "education", "{edu}",\n'
        s_predit_sql += f'            "default1", "{defa}", "balance", "{bal}",\n'
        s_predit_sql += f'            "housing", "{housing}", "loan", "{loan}",\n'
        s_predit_sql += f'            "contact", "{contact}", "day", "{day}",\n'
        s_predit_sql += f'            "month", "{month}", "duration", "{dura}",\n'
        s_predit_sql += f'            "campaign", "{camp}", "pdays", "{pdays}",\n'
        s_predit_sql += f'            "previous", "{prev}", "poutcome", "{outcome}"),\n'
        s_predit_sql += f'            "{model}", NULL)) prediction;'

        return s_predit_sql

    def execute_automl_predict(self, query):

        if len(query) < 1: 
            message = "You must make the prediction query for new data"
            logging.exception(message)
            raise gr.Info(message)

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        o_resp, exec_time = mydb.execute_pd_query(query)

        mydb.disconnect()

        o_resp = o_resp.to_dict()
        #print(o_resp)
        o_resp = o_resp['prediction'][0]
        o_resp = json.loads(o_resp)
        #print(o_resp)
    
        s_predict = o_resp['ml_results']['predictions']['y']
        s_proba_yes = o_resp['ml_results']['probabilities']['yes']
        s_proba_no = o_resp['ml_results']['probabilities']['no']

        restxt =  f'1. prediction: {s_predict}\n'
        restxt += f'2. probability(yes): {s_proba_yes}\n'
        restxt += f'3. probability(no): {s_proba_no}'

        resplot = gr.BarPlot(
            pd.DataFrame({ "Loan": ["yes", "no"], 
                  "Probability": [(s_proba_yes*100), (s_proba_no*100)]}),
            x="Loan",
            y="Probability",
            title="Probability of loan"
        )

        return restxt, resplot

    def load_llm_model(self, model):

        if not self.check_hw_node():
            raise gr.Error("Firstly, you must setup the heatwave node")        

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        o_out, o_resp, exec_time = mydb.execute_callproc('sys.ML_MODEL_LOAD', 
                   (f'{model}', 'null'))

        #print('output:', o_out)

        mydb.disconnect()

        return f'{model} LLM loaded : {o_resp}'

    def load_rag_doc(self, ragdb, ragtable, docurl):

        if len(ragdb) < 1 or len(ragtable) < 1 or len(docurl) < 1:
            raise gr.Error("You must input db, table, object-document url")

        _, file_format = os.path.splitext(docurl)
        file_format = file_format.split('.')[1]
        file_format = file_format.lower()

        s_input_list =  '[{ "db_name":"' + ragdb + '", ' 
        s_input_list += '   "tables": [{ '
        s_input_list += f'      "table_name": "{ragtable}", '
        s_input_list += '       "engine_attribute": { '
        s_input_list += '           "dialect": {"format": "'+file_format+'", "ocr": true, "language": "ko"}, '
        s_input_list += '           "file": [{"par":"' + docurl + '"}] } }] }]'

        s_options = {"mode": "normal"}

        if not self.check_hw_node():
            raise gr.Error("Firstly, you must setup the heatwave node")

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        o_out, _, exec_time = mydb.execute_callproc('sys.heatwave_load', 
                            (s_input_list, json.dumps(s_options)))
        mydb.disconnect()

        obj_ragtable = self.get_rag_table()

        return [ o_out, gr.Dropdown(obj_ragtable, label='vector db', 
                        value=obj_ragtable[0], interactive=True) ]

    def reload_vector_table(self):

        obj_ragtable = self.get_rag_table()

        return gr.Dropdown(obj_ragtable, label='vector db', 
                        value=obj_ragtable[0], interactive=True)

    def get_response(self, message, history, ragenable, ragmodel, ragdb, ragtable):

        o_resp=''

        if ragmodel == None or len(ragmodel) < 1:
            raise gr.Error("You have to load LLM")

        if len(message) < 1:
            return "You must input your question in chat!!"

        if ragmodel != "llama3-8b-instruct-v1":
            message = self.tranlat.translate(message, dest="en")
            message = message.text

        #print(history)

        if ragenable: 
            if len(ragmodel) < 1 or len(ragdb) < 1 or len(ragtable) < 1:
                raise gr.Error("You must input model, db, table using vector store")

            s_model_info =   '{'
#            s_model_info += f' "vector_store":["{ragdb}.{ragtable}"], ' #vector_store or schema
            s_model_info +=  ' "n_citations": 5, '
            s_model_info += f' "schema":["{ragdb}"], '
            s_model_info +=  ' "distance_metric": "COSINE", '
            s_model_info +=  ' "model_options": { '
            s_model_info +=  '    "temperature": 0.3, '
            s_model_info +=  '    "repeat_penalty": 1, '
            s_model_info +=  '    "top_p": 0.2, '
            s_model_info +=  '    "max_tokens": 1000, '
            if ragmodel == "llama3-8b-instruct-v1":
                s_model_info += ' "language": "ko", '
            s_model_info += f'    "model_id": "{ragmodel}" '
            s_model_info +=  ' } }'

            s_prompt_sql =  "SELECT JSON_UNQUOTE(JSON_EXTRACT(@output, '$.text')) as response;"
        else:
            s_prompt_sql =  f'select JSON_PRETTY(sys.ML_GENERATE("{message}", \n' 
            s_prompt_sql += f'JSON_OBJECT("task", "generation", \n' 
            if ragmodel == "llama3-8b-instruct-v1":
                s_prompt_sql += f'"language", "ko", \n'
            s_prompt_sql += f'"model_id", "{ragmodel}"))) as response;'

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        if ragenable:
            print(s_model_info)
            o_out, o_resp, exec_time = mydb.execute_callproc('sys.ML_RAG',
                    (message, ('output', 'CHAR'), s_model_info))
            #print('call ml_rag:', o_resp[1])
            o_resp = json.loads(o_resp[1])
            o_resp = o_resp['text']
        else:
            o_resp, exec_time = mydb.execute_pd_query(s_prompt_sql)
            o_resp = json.loads(o_resp.iloc[0]['response'])
            #print('chat resp:',o_resp['text'])
            o_resp = o_resp['text']

        mydb.disconnect()

        if ragmodel != "llama3-8b-instruct-v1":
            message = self.tranlat.translate(o_resp, dest="ko")
            message = message.text
        else :
            message = o_resp

        return message

    def get_rag_table(self):

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        if mydb.is_connect() != True:
            return []

        try:
            o_resp, exec_time = mydb.execute_pd_query(self.ragtable1)
        except mysqld.Error as err:
            logging.exception(str(err))
            raise gr.Error(err)
        
        #print(o_resp[o_resp.columns[0]].to_list(), ":", type(o_resp))

        mydb.disconnect()

        return o_resp[o_resp.columns[0]].to_list()

    def search_vector(self, ragdb, ragtable, search_word):

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        embeding_query = f'select sys.ML_EMBED_ROW("{search_word}", NULL) into @query_embedding;'

        try:
            exec_time = mydb.execute_query(embeding_query)
        except mysqld.Error as err:
            logging.exception(str(err))
            raise gr.Error(err)

        #embeding_vector = o_resp[o_resp.columns[0]][0]

        search_query =  'select segment, '
        search_query += f"(1-distance(segment_embedding, @query_embedding, 'COSINE')) as similr "
        search_query += f'from {ragdb}.{ragtable} '
        search_query += 'order by similr '
        search_query += 'desc limit 3;'

        try:
            o_resp, exec_time = mydb.execute_pd_query(search_query)
        except mysqld.Error as err:
            logging.exception(str(err))
            raise gr.Error(err)

        mydb.disconnect()

        #return tabulate(o_resp, headers='keys', tablefmt='grid')
        return o_resp.to_html(columns=['similr', 'segment'])

    def write_resultsql(self, sql, parurl, fileformat):

        s_write_sql = sql.rstrip(';') 
        s_write_sql += ' into outfile with parameters '
        s_write_sql += ' \'{"file":[{"par":"'+parurl+'","prefix":"out"}], '
        if fileformat == "csv":
            s_write_sql += '"dialect": {"has_header":true, "format": "'+fileformat+'"}}\';'
        else: 
            s_write_sql += '"dialect": {"format": "'+fileformat+'"}}\';'

        #print(s_write_sql)

        mydb = MySQLdb(self.o_server[0], 3306, self.o_server[1], self.o_server[2])

        try:
            o_resp, exec_time = mydb.execute_affected_query(s_write_sql)
        except mysqld.Error as err:
            logging.exception(str(err))
            raise gr.Error(err)

        mydb.disconnect()

        return ( str(o_resp) + " rows wrote" )

    def make_webui(self):
        self.iface = None

        obj_ragtable = self.get_rag_table()

        with gr.Blocks(css='footer{display:none !important}', title='Heatwave') as self.iface:
            with gr.Row():
                gr.Markdown(f'## **HeatWave Demo Site in terms of tech (version:{ compatibility_heatwave })**')
            with gr.Tab('OVERVIEW'):
                with gr.Row():
                    gr.Markdown('> ### **Heatwave service overall architecture (No ETL, NO Expertise, No extra Cost, No security issue, very Simple)**')
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    gr.Image(label='Heatwave architecture', show_label=False,
                            value='config/heatwave_overall.jpg', type='filepath',
                            width='300px', scale=0.9)
                with gr.Row():
                    gr.Markdown('> ### **Heatwave 데모영상**: [here](https://objectstorage.ap-seoul-1.oraclecloud.com/p/C9RO87AQB7-mbnMhyyIeGq5Jnm6z5-wLvOVH6B5eFw6OhYqeAUF99nhIAxHCokrk/n/apaccpt01/b/docs/o/heatwave_demo.pptx)')
                with gr.Row():
                    gr.Markdown('> ### **The cost estimator site of Heatwave service**: [here](https://www.oracle.com/cloud/costestimator.html)')
            with gr.Tab('OLTP'):
                with gr.Row():
                    gr.Markdown(
                    """
                    > ### Heatwave MySQL는 기본적으로 **MySQL Enterprise Endition** 을 기반으로 함


                    | 주요기능         | 주요 기능에 대한 설명                                     |
                    |------------------|-----------------------------------------------------------|
                    | MySQL Masking    | 민감한 데이터에 대한 비식별화 및 익명화 지원              |
                    | MySQL Audit      | 사용자의 활동 감사 및 규정 준수 확인                      |
                    | MySQL Thread Pool| 대용량 트랙픽(병렬세션) 발생시 thread 기반 안정적 처리    |
                    | MySQL HA         | Innodb cluster 기반 이중화 서비스                         |
                    | MySQL TDE        | Oracle 관리Key 기반에 물리적 데이터 암호화 제공           |
                    | MySQL Encryption | 공개 또는 비밀키 기반에 대칭/비대칭 데이터 암호화 제공    |
                    | MySQL Autopilot  | 머신 러닝 기반으로 서비스 advisor, automated 기능 제공<br> (System setup,Data load,Failure handling,Query exectuion) |
                    | MySQL Autoindex  | Perfomance schema에 SQL 기반 secondary index 생성/삭제제안|
                    """
                    )
                with gr.Row():
                    gr.Markdown('> ### **Heatwave HA or Dr architecture**')
                with gr.Row():
                    gr.Image(label='Heatwave HA and Dr', show_label=False,
                            value='config/heatwave_ha_dr.jpg', type='filepath',
                            width='400px')
            with gr.Tab('OLAP(+)') as self.gtab1:
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown('> **MySQL Info (OLTP)**')
                        self.mysip = gr.Textbox(label='DB IP', value='127.0.0.1', interactive=True)
                        self.myport = gr.Number(label='DB Port', value=3306, interactive=True)
                        self.myuser = gr.Textbox(label='DB Username', value='admin', interactive=True)
                        self.mypass = gr.Textbox(label='DB Password', type='password', 
                                value='Welcome#1', interactive=True)
                    with gr.Column(scale=1):
                        gr.Markdown('> **HeatWave Info (OLAP)**')
                        self.htip = gr.Textbox(label='DB IP',value=self.o_server[0],interactive=True)
                        self.htport = gr.Number(label='DB Port',value=3306,interactive=True)
                        self.htuser = gr.Textbox(label='DB Username', 
                                                value=self.o_server[1],
                                                interactive=True)
                        self.htpass = gr.Textbox(label='DB Password', type='password',
                                                value=self.o_server[2],
                                                interactive=True)                    
                with gr.Row():
                    with gr.Accordion('See the analytic query!', open=False):
                        gr.Code(self.olapquery1)
                with gr.Row(equal_height=True):
                    with gr.Column():
                        self.mybtn = gr.Button('execution')
                    with gr.Column():
                        self.htbtn = gr.Button('execution')
                with gr.Row() as self.res:
                    with gr.Column():
                        self.myresbox = gr.Textbox(label='mysql response')
                    with gr.Column():
                        self.htresbox = gr.Textbox(label='heatwave response')
                with gr.Row():
                    with gr.Column(scale=1):
                        self.myexpl = gr.Textbox(label='MySQL explain plan')
                    with gr.Column(scale=1):
                        self.htexpl = gr.Textbox(label='heatwave explain plan')
                with gr.Row(equal_height=True) as self.output :
                    with gr.Column(scale=1):
                        self.plotmysql = gr.BarPlot(label='MySQL')
                    with gr.Column(scale=1):
                        self.plotheatw = gr.BarPlot(label='HeatWave')
            with gr.Tab('Lakehouse(+, >512GB)') as self.gtab2:
                gr.Markdown('> ### **View bank csv files in objct storage**')
                gr.Markdown('---')
                o_files = self.o_fs.ls(self.options.bucket + '@' + self.options.namespace, refresh=True)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        self.chkgrpobj = gr.Radio(o_files, 
                                             label='csv files(obj)', 
                                             info='files list in oci object storage')
                        self.rlobjbtn = gr.Button('Object Reload')
                    with gr.Column(scale=1):
                        self.fieldsep = gr.Dropdown([',',';','tab'], 
                                             label='field delimiter', value=';',
                                             info='select field delimiter')
                with gr.Row(equal_height=True):
                    #self.viewcts = gr.Textbox(value='click the object files',
                    #    label='csv contents view', lines=10)
                    self.viewcts = gr.Markdown(height='250px')
                with gr.Row():
                    self.loadcsv = gr.Markdown('> ### **Load bank csv data into heatwave**')
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    with gr.Column(scale=1):
                        self.dbname = gr.Textbox(label='db',value='AutoML',interactive=True)
                    with gr.Column(scale=1):
                        self.table = gr.Textbox(label='table', interactive=True)
                    with gr.Column(scale=3):
                        self.parurl = gr.Textbox(label='pre-authenticated url in object storage', 
                                value=obj_par_url.get('lkread'), interactive=True)
                with gr.Row():
                    self.genloadbtn = gr.Button('Make Query')
                with gr.Row():
                    self.loadcode = gr.Code(label='Auto load query', interactive=True)
                with gr.Row():
                    self.execautoload = gr.Button('Apply Query to HW(Make table)')
                with gr.Row():
                    self.autoresult = gr.Textbox(label='Result of autoload')
                with gr.Row():
                    gr.Markdown('> ### **Write analytic query into object storage**')
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    with gr.Accordion('See write back query!', open=False):
                     self.writesql = gr.Code(self.lkwrite1)
                with gr.Row():
                    with gr.Column(scale=1):
                        self.fileformat = gr.Dropdown(['csv','parquet'],
                                             label='file format', value='csv',
                                             info='select file format for write', 
                                             interactive=True)
                    with gr.Column(scale=4):
                        self.writeparurl = gr.Textbox(label='par url for write', 
                                value=obj_par_url.get('lkwrite'), interactive=True)
                with gr.Row():
                    self.genwritebtn = gr.Button('Write back of query result')
                with gr.Row():
                    self.writeresult = gr.Textbox(label='Result of Write')

            with gr.Tab('AutoML(+)') as self.gtab3:
                gr.Markdown('> ### **Train Model(based on data imported previous lakehouse)**') 
                gr.Markdown('---')
                with gr.Row():
                    self.amtab = gr.Textbox(label='Main table',value='AutoML.bankfull', 
                                interactive=True)
                with gr.Row():
                    with gr.Column():
                        self.amtrainbtn = gr.Button('Make Train query')
                    with gr.Column():
                        self.amtrainexbtn = gr.Button('Apply query to HW')
                with gr.Row():
                    with gr.Column():
                        self.amtraincode = gr.Code(label='AutoML train query', 
                                interactive=True)
                    with gr.Column():
                        self.amtrainresult = gr.Textbox(label='trained model name',lines=3)
                with gr.Row():
                    gr.Markdown('> ### **Load model trained to Heatwave memory**')
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    with gr.Column():
                        self.amloadmdbtn = gr.Button('Load model to HW')
                    with gr.Column():
                        self.amloadresult = gr.Textbox(label='load result')
                with gr.Row():
                    gr.Markdown('> ### **Predict loan or not for new bank data based on Model**')
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    self.age=gr.Number(label='age',value=30,interactive=True)
                    self.job=gr.Dropdown(['housemaid','unknown','blue-collar',
                       'student','management','technician','retired','admin',
                       'self-employed','services','entrepreneur','unemployed'],
                       value='services', label='job', interactive=True)
                    self.marital=gr.Dropdown(['married','single','divorced'],
                       label='marital',value='single',interactive=True) 
                    self.edu=gr.Dropdown(['secondary','unknown','tertiary','primary'],
                       label='education', value='primary', interactive=True)
                with gr.Row():
                    self.defa=gr.Dropdown(['yes','no'],label='default1',
                       value='yes',interactive=True)
                    self.bal=gr.Number(label='balance',value=1362,interactive=True)
                    self.housing=gr.Dropdown(['yes','no'],label='housing',
                       value='yes',interactive=True)
                    self.loan=gr.Dropdown(['yes','no'],label="loan",
                       value='yes',interactive=True)
                with gr.Row():
                    self.contact=gr.Dropdown(['telephone','unknown','cellular'],
                       label='contact', value='telephone', interactive=True)
                    self.day=gr.Number(label='day',value=15,minimum=1,
                       maximum=31,interactive=True)
                    self.month=gr.Dropdown(['jan','feb','mar','apr','may','jun','jul',
                       'aug','sep','oct','nov','dec'],label='month',
                       value='jul',interactive=True)
                    self.dura=gr.Number(label='duration',value=258,minimum=0,
                       interactive=True)
                with gr.Row():
                    self.camp=gr.Number(label='campaign',value=3,minimum=1,
                       maximum=365,interactive=True)
                    self.pdays=gr.Number(label='pdays',value=40,interactive=True)
                    self.prev=gr.Number(label='previous',value=0.6,minimum=0,
                       interactive=True)
                    self.poutcome=gr.Dropdown(['other','success','failure','unknown'],
                       label='poutcome', value='success', interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        self.ampredbtn=gr.Button('Make prediction query')
                    with gr.Column(scale=2):
                        self.ampredexebtn=gr.Button('Predict new data')
                with gr.Row():
                    with gr.Column():
                        self.ampredcode=gr.Code(label='Predict query',lines=12,interactive=True)
                    with gr.Column():
                        self.ampredresult=gr.Textbox(label='result of prediction',lines=8)
                    with gr.Column():
                        self.ampredgh=gr.BarPlot(label='probability')

            with gr.Tab('GenAI(+,>512GB)') as self.gtab4:
                gr.Markdown('> ### **Workflow of GenAI**')
                gr.Markdown('---')
                with gr.Row():
                    gr.Image(label='GenAI flow', show_label=False,
                            value='config/genai.jpg', type='filepath', scale=0.7)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('> ### **Choose embeded LLM and Load**') 
                with gr.Row():
                    gr.Markdown('---')
                with gr.Row():
                    with gr.Column():
                        self.modelsep=gr.Dropdown(obj_model_list,
                                             label='Embeded LLM',value=obj_model_list[0],
                                             info='select LLM',interactive=True)
                        self.llmbtn = gr.Button('Load selected LLM')
                    with gr.Column():
                        self.llmresult = gr.Textbox(label='result of LLM', lines=5)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('> ### **Build vector store(for rag)**')
                with gr.Row():
                        gr.Markdown('---')
                with gr.Row():
                    with gr.Column(scale=3):
                        self.ragdoc = gr.Textbox(label='doc pre-authenticated file url', 
                                value=obj_par_url.get('genai'), lines=5)
                    with gr.Column(scale=1):
                        with gr.Group():
                            self.ragdb=gr.Textbox(label='vector db',value='ragdb',
                                interactive=False)
                            self.ragtable=gr.Textbox(label='vector table',value='vectable1',
                                interactive=True)
                with gr.Row():
                    self.ragloadbtn = gr.Button('Apply to vector store (auto parallel load)')
                with gr.Row():
                    with gr.Column():
                        self.ragloadresult = gr.Textbox(label='load result of Rag',lines=5)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('> ### **Search vector and Ask the question to GenAI**')
                        gr.Markdown('---')
                with gr.Row():
                    with gr.Column(scale=2):
                        self.ragendb = gr.Textbox(label='vector db', value='ragdb',
                                interactive=False)
                    with gr.Column(scale=2):
                        if len(obj_ragtable) > 0:
                            self.ragentable = gr.Dropdown(obj_ragtable,
                                                label='vector table', value=obj_ragtable[0],
                                                interactive=True)
                        else:
                            self.ragentable = gr.Dropdown(obj_ragtable,
                                                label='vector table', value=None,
                                                interactive=True)
                    with gr.Column(scale=1):
                        self.ragldtabbtn = gr.Button('table reload')
                with gr.Row():
                    with gr.Column(scale=4):
                        self.vectorsearchn = gr.Textbox(label='vector search', interactive=True)
                    with gr.Column(scale=1):
                        self.vectorbtn = gr.Button('search')
                with gr.Row():
                    self.vectorresult = gr.HTML(label='vector search result (only get one with top similarity)')
                with gr.Row():
                    self.ragenable = gr.Checkbox(label='rag enable')
                with gr.Row():
                    with gr.Column():
                        gr.ChatInterface(self.get_response, 
                                additional_inputs=[self.ragenable,self.modelsep,
                                    self.ragendb,self.ragentable])
            '''
            with gr.Tab('GenAI(similar search)') as self.gtab5:
                with gr.Row():
                    embeded_iframe=f"""
                    <iframe
                        src="{similar_url}"
                        frameborder="0"
                        referrerpolicy="unsafe-url"
                        browsingtopics>
                    </iframe>
                    """
                    gr.HTML(embeded_iframe)
            '''

            self.gtab1.select(None, scroll_to_output=True)
            self.mybtn.click(self.execute_olap, \
                            [gr.Textbox(value='mysql', visible=False), 
                                self.mysip, self.myport, self.myuser, self.mypass], \
                            [self.myresbox, self.myexpl, self.plotmysql])
            self.htbtn.click(self.execute_olap, \
                            [gr.Textbox(value='heatwave', visible=False), 
                                self.htip, self.htport, self.htuser, self.htpass], \
                            [self.htresbox, self.htexpl, self.plotheatw])
            self.chkgrpobj.select(self.get_objcontents, 
                            [self.chkgrpobj, self.fieldsep],
                            [self.viewcts, self.loadcsv, self.table])
            self.rlobjbtn.click(self.load_objfiles, None, 
                            [self.chkgrpobj,self.viewcts,self.parurl,self.loadcode,
                             self.autoresult])
            self.fieldsep.change(self.get_objcontents,
                            [self.chkgrpobj, self.fieldsep],
                            [self.viewcts, self.loadcsv])
            self.genloadbtn.click(self.make_autoload_query, 
                            [self.dbname, self.table, self.parurl, self.fieldsep],
                            [self.loadcode])
            self.execautoload.click(self.execute_autoload, 
                            [self.dbname, self.table, self.parurl, self.fieldsep],
                            self.autoresult)
            self.amtrainbtn.click(self.make_automl_train, self.amtab, self.amtraincode)
            self.amtrainexbtn.click(self.execute_automl_train, 
                            self.amtab, self.amtrainresult)
            self.amloadmdbtn.click(self.load_automl_model, 
                            self.amtrainresult, self.amloadresult)
            self.ampredbtn.click(self.make_automl_predict, [self.age, self.job, self.marital,
                            self.edu,self.defa,self.bal,self.housing,self.loan,self.contact,
                            self.day,self.month,self.dura,self.camp,self.pdays,self.prev,
                            self.poutcome,self.amtrainresult],self.ampredcode)
            self.ampredexebtn.click(self.execute_automl_predict,self.ampredcode,
                            [self.ampredresult,self.ampredgh])
            self.llmbtn.click(self.load_llm_model, [self.modelsep], self.llmresult)
            self.ragloadbtn.click(self.load_rag_doc, [self.ragdb,self.ragtable,self.ragdoc],
                            [self.ragloadresult, self.ragentable])
            self.vectorbtn.click(self.search_vector, 
                            [self.ragendb,self.ragentable, self.vectorsearchn], 
                            [self.vectorresult])
            self.ragldtabbtn.click(self.reload_vector_table, None, self.ragentable)
            self.genwritebtn.click(self.write_resultsql, 
                            [self.writesql, self.writeparurl, self.fileformat],
                            self.writeresult)

        self.iface.launch(server_name="0.0.0.0", server_port=8000, show_error=True, 
                          auth=("admin", "pass1234##"),ssl_certfile="cert.pem", 
                          ssl_keyfile="key.pem", ssl_verify=False,
                          favicon_path="config/heatwave_icon.png")

        #self.iface.launch(server_name="0.0.0.0", server_port=8000, show_error=True,
        #                 auth=("admin", "pass1234##"), favicon_path="config/heatwave_icon.png")

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
