import sys
import os
import csv
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import googlemaps
import json
import re

#Gets addresses from GMAPS API

pd.set_option('display.max_columns', None) 
attributes = ['LON', 'LAT', 'NUMBER', 'STREET', 'CITY', 'REGION', 'POSTCODE']


gmaps = googlemaps.Client(key='Enter your own key')


abrv_dict = {"ALLEY":"ALLEY", "AVENUE":"AVE", "BOULEVARD":"BLVD", "BRIDGE":"BR", "CANYON":"CYN", "CENTER":"CTR", "CIRCLE":"CIR", "COURT":"CT", "CRESCENT":"CRES", "DRIVE":"DR", "EAST":"E", "EXPRESSWAY":"EXPWY", "FREEWAY":"FWY", "HIGHWAY":"HWY", "HILL":"HILL", "INFORMATION":"INFO", "INTERNATIONAL":"INTL", "ISLAND":"ISL", "JUNCTION":"JCT", "LANE":"LN", "LAKE":"LK", "LOOP":"LP", "MOUNT":"MT", "MOUNTAIN":"MTN", "NORTH":"N", "NATIONAL":"NATL", "NORTHEAST":"NE", "NORTHWEST":"NW", "PARKWAY":"PKWY", "PLACE":"PL", "PLAZA":"PLZ", "PORT":"PT", "RIVER":"R", "ROAD":"RD", "ROUTE":"RTE", "SQUARE":"SQ", "SOUTH":"S", "SOUTHEAST":"SE", "SOUTHWEST":"SW", "STREET":"ST", "TERRACE":"TER", "THRUWAY":"THWY", "TRAIL":"TR", "TURNPIKE":"TPK", "WAY":"WAY", "WEST":"W"}

data = pd.read_csv('countrywide.csv',encoding = 'utf-8',nrows=30, usecols = attributes)
data2 = pd.DataFrame(data = None, columns=list(data))


for i in range(len(data)):
	cur_street = data.loc[i,'STREET']
	for key in abrv_dict.keys():
		if key in cur_street:
			data2 = pd.concat([data2,data.loc[[i]]])
			data2.loc[i,'STREET'] = cur_street.replace(key, abrv_dict[key])
			break

data = data2
data = data.dropna()
temp = data.NUMBER.str.contains(r'[A-Za-z\-]',regex = True)

data = data[~temp]


data = data.reset_index()
final_data = pd.DataFrame(columns=['NUMBER', 'STREET', 'CITY', 'REGION', 'POSTCODE', 'ADDRESS'])
data_res = []


for ind in range(len(data)):
	reverse_geocode = gmaps.reverse_geocode((data.loc[ind]['LAT'],data.loc[ind]['LON']))
	row = []

	for i in range(len(reverse_geocode)):
		if data.loc[ind]['STREET'].split(' ', 1)[0].capitalize() in reverse_geocode[i]["formatted_address"] and (data.loc[ind]['CITY'].split(' ', 1)[0].capitalize() in reverse_geocode[i]["formatted_address"]) and (data.loc[ind]['REGION'].split(' ', 1)[0] in reverse_geocode[i]["formatted_address"]) and (str(int(data.loc[ind]['POSTCODE'])) in reverse_geocode[i]["formatted_address"]) and ("Unit" not in reverse_geocode[i]["formatted_address"]) and (bool(re.search(r'(^\d+)', reverse_geocode[i]["formatted_address"]))) and (bool(re.search(r'(^((?!\d\-).)*$)',reverse_geocode[i]["formatted_address"]))) and (bool(re.search(r'(^((?!\d\-).)*$)', str(data.loc[ind]['NUMBER'])))):

			row.append(data.loc[ind]['NUMBER'])
			row.append(data.loc[ind]['STREET'])
			row.append(data.loc[ind]['CITY'].title())
			row.append(data.loc[ind]['REGION'])
			row.append((str(int(data.loc[ind]['POSTCODE']))))
			row.append(re.sub(r'(^\d+)',str(data.loc[ind]['NUMBER']),reverse_geocode[i]["formatted_address"]).replace(",", ""))
			break

	if len(row) != 0:
		data_res.append(row)

final_data = pd.DataFrame(data_res, columns=['NUMBER', 'STREET', 'CITY', 'REGION', 'POSTCODE', 'ADDRESS'])
print(final_data)
final_data.to_csv("final_data.csv", encoding='utf-8', index=False)
