import pickle
import os.path
import glob as glob

from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload 
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def download_from_drive_folder(country, folder_id, scopes=['https://www.googleapis.com/auth/drive.readonly']):
    """Download content from google drive folder containing google earth engine images
    :param folder_id: folder id, retrievable from the url
    :type folder_id: str
    :param scopes: _description_, defaults to ['https://www.googleapis.com/auth/drive.readonly']
    :type scopes: list, optional
    """
    creds = None
    if os.path.exists(Path('../../../conf') / 'token.pickle'):
        with open(Path('../../../conf') / 'token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                glob.glob(str(Path('../../../conf') / '*.json'))[0], scopes
                )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(Path('../../../conf') / 'token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=10,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        items = results.get('files', [])
        items = [d for d in items if country.lower() in d['name']]
        if not items:
            print('No files found.')
        else:
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
                file_id = item['id']
                request = service.files().get_media(fileId=file_id)
                with open(Path('../../../data/external/gee') / item['name'], 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break