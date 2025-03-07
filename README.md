# Reading CSV Files from Google Drive

This Python script demonstrates how to read CSV files from Google Drive using the Google Drive API and pandas.

## Setup Instructions

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Google Drive API:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Drive API for your project
   - Create credentials (OAuth 2.0 Client ID) for a desktop application
   - Download the credentials and save them as `credentials.json` in the same directory as the script

3. Get your file ID:
   - Open your CSV file in Google Drive
   - Get the file ID from the sharing URL
   - The file ID is the long string of characters between `/d/` and `/view` in the URL
   - Example URL: `https://drive.google.com/file/d/YOUR_FILE_ID_HERE/view`

## Usage

1. Replace `YOUR_FILE_ID_HERE` in the `main()` function with your actual Google Drive file ID.

2. Run the script:
   ```bash
   python read_google_drive_csv.py
   ```

3. On first run:
   - A browser window will open asking you to authenticate with your Google account
   - Grant the requested permissions
   - The authentication tokens will be saved locally in `token.pickle` for future use

## Features

- Authenticates with Google Drive API
- Downloads CSV file content
- Converts CSV data to pandas DataFrame
- Shows download progress
- Handles errors gracefully
- Caches authentication tokens for future use

## Notes

- The script requires internet connectivity to access Google Drive
- Make sure your CSV file is accessible with the Google account you use for authentication
- The `credentials.json` file should never be shared or committed to version control 