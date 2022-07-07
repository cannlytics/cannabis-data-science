"""
CCRS Utility Functions | Cannlytics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/11/2022
Updated: 4/11/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
import os
import zipfile


# TODO: Download the latest CCRS data from box.
# https://lcb.app.box.com/s/7pi3wqrmkuo3bh5186s5pqa6o5fv8gbs
# from boxsdk import DevelopmentClient# pip install boxsdk


# Alternative: Download CCRS from Cannlytics.
def download_latest_ccrs_zip(filename):
    """Download the latest CCRS data as a zipped folder."""
    # file_id = url.split('/')[-1]
    # client = DevelopmentClient()
    # with open(filename, 'wb') as open_file:
    #     client.file(file_id).download_to(open_file)
    #     open_file.close()
    # TODO: Allow Cannlytics subscribers to download.
    # or implement a token system (Cannlytics DataCoins | CDC)
    raise NotImplementedError


# def unzip_files(dir_name, extension='.zip'):
#     """Unzip all files in a specified folder."""
#     os.chdir(dir_name) # change directory from working dir to dir with files
#     for item in os.listdir(dir_name): # loop through items in dir
#         if item.endswith(extension): # check for ".zip" extension
#             file_name = os.path.abspath(item) # get full path of files
#             zip_ref = zipfile.ZipFile(file_name) # create zipfile object
#             zip_ref.extractall(dir_name) # extract file to dir
#             zip_ref.close() # close file
#             os.remove(file_name) # delete zipped file


def unzip_files(_dir, extension='.zip'):
    """Unzip all files in a specified folder.
    Author: nlavr https://stackoverflow.com/a/69101930
    License: CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
    """
    for item in os.listdir(_dir):  # loop through items in dir
        abs_path = os.path.join(_dir, item)  # absolute path of dir or file
        if item.endswith(extension):  # check for ".zip" extension
            file_name = os.path.abspath(abs_path)  # get full path of file
            zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
            zip_ref.extractall(_dir)  # extract file to dir
            zip_ref.close()  # close file
            os.remove(file_name)  # delete zipped file
        elif os.path.isdir(abs_path):
            unzip_files(abs_path)  # recurse this function with inner folder


# Optional: Create a function to parse analyses.
# for test in list(lab_results.TestName.unique()):
#     try:
#         test_type = test.split(' -')[0]
#         analysis = snake_case(test.split(' - ')[1].split('(')[0].strip())
#         units = test.split('(')[1].replace(')', '')
#         print (f"'{test}': {{'key': '{analysis}', 'units': '{units}'}},")
#     except:
#         # print('Parse error:', test)
#         pass


if __name__ == '__main__':

    # Create a place for your data to live.
    data_dir = 'D:\\data\\washington'

    # FIXME: Download the latest zipped datasets programattically.
    # download_latest_ccrs_zip('../../.datasets/cannlytics/ccrs.zip')
    # dropbox_file_download(
    #     None,
    #     'https://lcb.app.box.com/s/7pi3wqrmkuo3bh5186s5pqa6o5fv8gbs',
    #     '../../.datasets/cannlytics',
    # )

    # Get the filename.
    latest_file = 'CCRS PRR All Data Up To 3-12-2022'

    # # Unzip the datasets.
    # unzip_files(data_dir)

    # # Unzip each dataset.
    # unzip_files(f'{data_dir}/{latest_file}')

    # TODO: Test unzipping the entire folder.
    unzip_files(f'{data_dir}/{latest_file}', extension='.zip')
