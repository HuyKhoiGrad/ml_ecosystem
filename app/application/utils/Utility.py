# encoding: utf-8
# Description: implement some common utilities
import sys, os

import json
import re, time
import subprocess, glob
import string
import random
import calendar
import hashlib
import re
import hashlib

from datetime import datetime, timedelta, date
from inspect import stack
import pendulum
from typing import List

import yaml
from xml.dom import ValidationErr

# time is greater than 7 hours
MINUS_TIME = 7 * 60 * 60 * 1000
LOCAL_TZ = pendulum.timezone("Asia/Ho_Chi_Minh")

ConvertNone2Empty = lambda value_object:"" if value_object is None else value_object

class Utilities:
	def __init__(self):
		pass
	
	@staticmethod
	def WriteErrorLog(strErrorMsg, oConfig):
		strFileLog	= "%s%s.%s" % (oConfig.ErrorLog, time.strftime("%Y%m%d", time.localtime()), "log")
		fnLog		= open(strFileLog, "a")
		try:
			timeAt		= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
			strMsg		= '[%s][%s][%s]\r\n' % (timeAt, "ERROR", strErrorMsg)
			fnLog.write(strMsg)
		except Exception as exc:
			strErrorMsg = 'Error: %s' % str(exc) # give a error message
			sys.stderr.write(strErrorMsg)
		finally:
			fnLog.close()

	@staticmethod
	def WriteDataLog(strInfoMsg, strType, oConfig):
		strFileLog	= "%s%s.%s" % (oConfig.DataLog, time.strftime("%Y%m%d", time.localtime()), "log")
		
		fLog = open(strFileLog, 'a')
		try:
			timeAt		= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
			strMsg		= '[%s][%s][%s]\r\n' % (timeAt, strType, strInfoMsg)
			fLog.write(strMsg)
		except Exception as exc:
			strErrorMsg = 'Error: %s\n' % str(exc) # give a error message
			sys.stderr.write(strErrorMsg)
		finally:
			fLog.close()

	# Read YAML files
	@staticmethod
	def loadYAMLConfig(configFile=None):

		#print('Run loadYAMLConfig')

		if os.path.isfile(configFile) and os.path.splitext(configFile)[1] == '.yaml':
			#print(f'File {configFile} existed!!!')
			with open(configFile,'r') as f:
				return yaml.safe_load(f)

		else:
			raise ValidationErr(f"File {configFile} is not existed")
			return None

 	# Check is correct datetime
	@staticmethod
	def is_correct_datetime(value) -> bool:
		if isinstance(value, datetime):
			return True
		if isinstance(value, int):
			return True
		return False

	# Update dirty datetime is correct datetime
	@staticmethod
	def update_dirty_datetime_data(source: dict, datetime_fields: List[str]):
		for field in datetime_fields:
			if field not in source:
				continue
			value = source[field]
			if isinstance(value, dict) and '$date' in value:
				correct_date = value['$date'] - MINUS_TIME
				source[field] = pendulum.from_timestamp( correct_date / 1000)
				continue

			if Utilities.is_correct_datetime(value):
				continue
			source[field] = None

	@staticmethod
	def get_current_time():
		return pendulum.now(tz=LOCAL_TZ)
	
	@staticmethod
	def convert_timestamp_without_tz(value: int):
		return pendulum.from_timestamp(value / 1000, tz=LOCAL_TZ)

	@staticmethod
	def checkFileExists (strPathFile):
		if os.path.exists(strPathFile):
			return True
		else:
			return False

	@staticmethod
	def generate_hashed_filename(filename)->tuple[str, str]:
		with open(filename, "rb") as file:
			# Read the content of the file in binary mode
			content = file.read()

		# Calculate the SHA256 hash of the file content
		sha256_hash = hashlib.sha256(content).hexdigest()

		# Append the hash to the original filename (excluding the file extension, if any)
		base_filename, file_extension = filename.rsplit(".", 1)
		hashed_filename = f"{base_filename}_{sha256_hash}.{file_extension}"
		file_size = os.path.getsize(filename)
		return hashed_filename, sha256_hash, file_size

	@staticmethod
	def getFileSize(strFullName):
		lFileSize = 0
		try:
			lFileSize = os.path.getsize(strFullName)
		except:
			lFileSize = 0
		finally:
			return lFileSize

	#***************************************************************
	# Function: ConvertDateTime
	# Description: Convert date time as format
	# Parameter: strTimeStamp, strFormat, strNewFormat
	# Resutl: string
	#***************************************************************
	@staticmethod
	def convertDateTime(strTimeStamp, strFormat, strNewFormat):
		strResult = ""
		try:
			dtTime 	  = datetime.strptime(strTimeStamp, strFormat)
			strResult = dtTime.strftime(strNewFormat)
		except:
			strResult = ""
		return strResult

	#***************************************************************
	# Function: ConvertTimeStamp
	# Description: Convert unix time to time stamp
	# Parameter: iClock, strFormat
	# Result: string
	#***************************************************************
	@staticmethod
	def convertTimeStamp(iClock, strFormat):
		strResult = ""
		if iClock == 0:
			return strResult

		try:
			oDatetime 	  = datetime.fromtimestamp(int(iClock))
			tupleDatetime = oDatetime.timetuple()
			strResult     = time.strftime(strFormat, tupleDatetime)
		except:
			strResult = ""
		return strResult
	
	#***************************************************************
	# Function: GetStringFieldValue
	# Description: Get string field value of bson
	#***************************************************************
	@staticmethod
	def getStringFieldValue(dRecord, strField, strException = ''):
		try:
			strResult = str(ConvertNone2Empty(dRecord[strField]))
		except:
			strResult = strException

		return strResult

	#***************************************************************
	# Function: GetIntFieldValue
	# Description: Get int field value of bson
	#***************************************************************
	@staticmethod
	def getIntFieldValue(dRecord, strField, iException = 0):
		try:
			iResult = int(dRecord[strField])
		except:
			iResult = iException

		return iResult

	#***************************************************************
	# Function: GetFloatFieldValue
	# Description: Get float field value of bson
	#***************************************************************
	@staticmethod
	def getFloatFieldValue(dRecord, strField, fException = 0.0):
		try:
			fResult = float(dRecord[strField])
		except:
			fResult = fException

		return fResult
	
	#***************************************************************
	# Function: GetListFieldValue
	# Description: Get List field value of bson
	#***************************************************************
	@staticmethod
	def getListFieldValue(dRecord, strField):
		try:
			arrResult = dRecord[strField]
		except:
			arrResult = []

		return arrResult

	@staticmethod
	def replaceElementInList(array, condition, new_dict):
		for i, item in enumerate(array):
			if condition(item):
				array[i] = new_dict
				break  # Stop iterating once the dictionary is replaced

	@staticmethod
	def appendNewElementInList(array, condition, new_dict):
		bFlag = True
		for i, item in enumerate(array):
			if condition(item):
				bFlag = False

		if bFlag:
			array.append(new_dict)
   
	@staticmethod
	def upsertElementInList(array, condition, new_dict):
		bFlag = True
		for i, item in enumerate(array):
			if condition(item):
				array[i] = new_dict
				bFlag = False
				break  # Stop iterating once the dictionary is replaced

		if bFlag:
			array.append(new_dict)
   
	@staticmethod
	def extractElementInList(array, condition) -> dict():
		dResult = dict()
		for i, item in enumerate(array):
			if condition(item):
				dResult = item
				break
		return dResult
		
	@staticmethod
	def checkInstanceExist(oResultSet):
		try:
			dFirstItem = oResultSet[0]
			return dFirstItem
		except:
			return False