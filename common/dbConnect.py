import os
import pymssql
from configparser import ConfigParser
config = ConfigParser()
config.read('{}/conf.ini'.format(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


lsn_dict = {'1': 'KOR', '6': 'KOR', '10': 'KOR',
            '5': 'MATH', '7': 'MATH', '11': 'MATH', 
            '3': 'ENG', '12': 'ENG', '16': 'ENG', 
            '8': 'HIST', '13': 'HIST', 
            '4': 'TAM', '9': 'TAM', '14': 'TAM',
            '15': 'LANG'}
    
def dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, ST_END_DIV, CMPLT_MSG):
    conn = pymssql.connect(server=config['DBConn']['server'], \
                        user=config['DBConn']['user'],\
                        password=config['DBConn']['password'],\
                        database=config['DBConn']['database'])
    cursor = conn.cursor()
    sql = 'INSERT INTO [dbo].[LOG_SYS_TXN](LOG_DTM, STEP_NM, PID, JOB_NM, ARG_VAL, LOG_MSG, ST_END_DIV, CMPLT_MSG)' + \
        ' VALUES(' + \
        'GETDATE()' + ', ' + "'" + \
        str('03. PYTHON') + "'" + ', ' + "'" + \
        str(os.getpid()) + "'" + ', ' + "'" + \
        str('OMR01_01_EXECUTE PYTHON') + "'" + ', ' + "'" + \
        'EXAM_CD : ' + str(EXAM_CD) + ', ' + 'CMPN_CD : ' + str(CMPN_CD) + ', ' + 'BSTOR_CD : ' + str(
        BSTOR_CD) + ', ' + 'OMR_MST_CD : ' + str(OMR_MST_CD) + ', '\
        + 'SERVER : ' + config['ML']['Number'] + "'" + ', ' + "'" + \
        str(LOG_MSG) + "'" + ', ' + "'" + \
        str(ST_END_DIV) + "'" + ', ' + "'" + \
        str(CMPLT_MSG) + "')"

    try:
        cursor.execute(sql)
        conn.commit()
    except pymssql.Error as e:
        print(e)
        print(sql)
        pass

    cursor.close()
    conn.close()


def dbReupld(fileNM):
    conn = pymssql.connect(server=config['DBConn']['server'], \
                        user=config['DBConn']['user'],\
                        password=config['DBConn']['password'],\
                        database=config['DBConn']['database'])
    cursor = conn.cursor()
    sql = "UPDATE [dbo].[LOG_FILE_CHG_TXN]" + \
        " SET REUPLD_YN = 'Y'" + \
        ' WHERE CHG_FILE_NM = '+str(fileNM[:-4])

    try:
        cursor.execute(sql)
        conn.commit()
    except pymssql.Error as e:
        print(sql)
        pass

    cursor.close()
    conn.close()


def getThreshold(BSTOR_CD, OMR_MST_CD):
    conn = pymssql.connect(server=config['DBConn']['server'], \
                        user=config['DBConn']['user'],\
                        password=config['DBConn']['password'],\
                        database=config['DBConn']['database'])
    cursor = conn.cursor()
    sql = 'SELECT ' + lsn_dict[OMR_MST_CD] + ' FROM TB_OMR_THRESHOLD_KICE WHERE BSTOR_CD = ' + BSTOR_CD

    try:
        cursor.execute(sql)
        row = cursor.fetchone()

    except pymssql.Error as e:
        print(sql)
        pass

    cursor.close()
    conn.close()

    return row[0]


def UpdateThreshold(OMR_MST_CD, BSTOR_CD, C):
    conn = pymssql.connect(server=config['DBConn']['server'], \
                        user=config['DBConn']['user'],\
                        password=config['DBConn']['password'],\
                        database=config['DBConn']['database'])
    cursor = conn.cursor()
    sql = 'UPDATE TB_OMR_THRESHOLD_KICE SET ' + lsn_dict[OMR_MST_CD] + '=' + str(C) + 'WHERE BSTOR_CD = ' + BSTOR_CD

    try:
        cursor.execute(sql)
        conn.commit()
    except pymssql.Error as e:
        print(sql)
        pass
    cursor.close()
    conn.close()


def GetLsnCd(OMR_MST_CD):
    conn = pymssql.connect(server=config['DBConn']['server'], \
                           user=config['DBConn']['user'], \
                           password=config['DBConn']['password'], \
                           database=config['DBConn']['database'])
    cursor = conn.cursor()
    
    if lsn_dict[OMR_MST_CD] == 'KOR':
        sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD = 22050'
    elif lsn_dict[OMR_MST_CD] == 'MATH':
        sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD = 22105'
    elif OMR_MST_CD == '9':
        sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD IN (22250, 22300)'
    elif OMR_MST_CD == '14':
        sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD IN (22250, 22300, 22330)'
    # elif lsn_dict[OMR_MST_CD] == 'TAM':
    #     sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD IN (22250, 22300, 22330)'
    elif lsn_dict[OMR_MST_CD] == 'LANG':
        sql = 'SELECT LSN_CD FROM TB_LSN_MAPPG WHERE LSN_GRP_CD = 22350'
        
    lsn_cd = []
    try:
        cursor.execute(sql)
        row = cursor.fetchone()
        while row:
            lsn_cd.append(row[0])
            row = cursor.fetchone()
    except pymssql.Error as e:
        print(sql)
        pass

    cursor.close()
    conn.close()

    return lsn_cd