# -----------------------------------------------------------
# initialization of database instances
# Firebase, Google Drive, SQLite
#
# (C) 2021-2024 Juan-Francisco Reyes, Cottbus, Germany
# Brandenburg University of Technology, Germany.
# Released under MIT License
# email pacoreyes.zwei@gmail.com
# -----------------------------------------------------------
from google.cloud import firestore
import gspread
from utils.utils import load_json_file
from oauth2client.service_account import ServiceAccountCredentials


""" Initialize Firestore """
# Initialize Firestore database instance
firestore_db = firestore.Client.from_service_account_json('credentials/firebase_credentials.json')


""" Initialize Google Sheets """
gdrive_scope = ["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
sheet_id_1 = load_json_file("credentials/gsheets_credentials.json")["sheet_id_1"]
sheet_id_2 = load_json_file("credentials/gsheets_credentials.json")["sheet_id_2"]
sheet_id_3 = load_json_file("credentials/gsheets_credentials.json")["sheet_id_3"]
sheet_id_4 = load_json_file("credentials/gsheets_credentials.json")["sheet_id_4"]
sheet_id_5 = load_json_file("credentials/gsheets_credentials.json")["sheet_id_5"]

cred = ServiceAccountCredentials.from_json_keyfile_name("credentials/gsheets_credentials.json", gdrive_scope)
client = gspread.authorize(cred)

spreadsheet_1 = client.open_by_key(sheet_id_1)  # Initialize the spreadsheet 1 (websites, user )
spreadsheet_2 = client.open_by_key(sheet_id_2)  # Initialize the spreadsheet 2 (dataset1)
spreadsheet_3 = client.open_by_key(sheet_id_3)  # Initialize the spreadsheet 3 (dataset1_seminar - dataset2)
spreadsheet_4 = client.open_by_key(sheet_id_4)  # Initialize the spreadsheet 4 (dataset3)
spreadsheet_5 = client.open_by_key(sheet_id_5)  # Initialize the spreadsheet 5 (dataset1_seminar - dataset1)
