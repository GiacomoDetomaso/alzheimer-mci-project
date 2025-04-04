#!/bin/bash
#
#================================================================
# download_oasis_freesurfer.sh
#================================================================
#
# Usage: ./download_oasis_freesurfer.sh <input_file.csv> <directory_name> <nitrc_ir_username>
# 
# Download Freesurfer files from OASIS3 or OASIS4 on NITRC IR and organize the files
#
# Required inputs:
# <input_file.csv> - A Unix formatted, comma-separated file containing a column for freesurfer_id 
#       (e.g. OAS30001_Freesurfer53_d0129)
# <directory_name> - A directory path (relative or absolute) to save the Freesurfer files to
# <nitrc_ir_username> - Your NITRC IR username used for accessing OASIS data on nitrc.org/ir
#       (you will be prompted for your password before downloading)
#
# This script organizes the files into folders like this:
#
# directory_name/OAS30001_MR_d0129/$FREESURFER_FOLDERS
#
#
# Last Updated: 6/27/2024
# Author: Sarah Keefe
#
#
unset module

# Authenticates credentials against NITRC and returns the cookie jar file name. USERNAME and
# PASSWORD must be set before calling this function.
#   USERNAME="foo"
#   PASSWORD="bar"
#   COOKIE_JAR=$(startSession)
startSession() {
    # Authentication to XNAT and store cookies in cookie jar file
    local COOKIE_JAR=.cookies-$(date +%Y%M%d%s).txt
    curl -k -s -u ${USERNAME}:${PASSWORD} --cookie-jar ${COOKIE_JAR} "https://www.nitrc.org/ir/data/JSESSION" > /dev/null
    echo ${COOKIE_JAR}
}

# Downloads a resource from a URL and stores the results to the specified path. The first parameter
# should be the destination path and the second parameter should be the URL.
download() {
    local OUTPUT=${1}
    local URL=${2}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie ${COOKIE_JAR} -o ${OUTPUT} ${URL}
}

# Downloads a resource from a URL and stores the results to the specified path. The first parameter
# should be the destination path and the second parameter should be the URL. This function tries to
# resume a previously started but interrupted download.
continueDownload() {
    local OUTPUT=${1}
    local URL=${2}
    curl -H 'Expect:' --keepalive-time 2 -k --continue - --cookie ${COOKIE_JAR} -o ${OUTPUT} ${URL}
}

# Gets a resource from a URL.
get() {
    local URL=${1}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie ${COOKIE_JAR} ${URL}
}

# Ends the user session.
endSession() {
    # Delete the JSESSION token - "log out"
    curl -i -k --cookie ${COOKIE_JAR} -X DELETE "https://www.nitrc.org/ir/data/JSESSION"
    rm -f ${COOKIE_JAR}
}

# The function check if the input string do not contain more than a single space between words
is_valid_format() {
    local INPUT_STRING="$1"

    # Ensure input does not start or end with a space and has only single spaces between words
    if [[ "$INPUT_STRING" =~ ^[a-zA-Z0-9]+( [a-zA-Z0-9]+)*$ ]]; then
        return 0  # Valid format
    else
        echo "Error: Input must be words separated by single spaces only." >&2
        return 1  # Invalid format
    fi
}

# Function to check if a word is in the list.
# The input of this function is the name of the folder.
is_valid_folder() {
    local FOLDER=$1
    for VALID in "${VALID_FOLDERS[@]}"; do
        if [[ "$FOLDER" == "$VALID" ]]; then
            return 0  # Found, return success
        fi
    done
    return 1  # Not found, return failure
}

# This function validate the user input folders.
# The input of this function is the user single spaced string with folders' names.
validate_folders() {
    local INPUT_STRING="$1"
    echo $INPUT_STRING

    if is_valid_format "$INPUT_STRING";
    then
        IFS=' ' read -r -a USER_FOLDERS <<< "$INPUT_STRING"  # Convert input string to array

        for FOLDER in "${USER_FOLDERS[@]}"; do
            if ! is_valid_folder "$FOLDER"; then
                echo "Error: '$FOLDER' is not in the valid list." >&2
                return 1 # error: folder not in the list
            fi
        done
    else 
        return 1 # error the folder contains spaces
    fi

    return 0
}

# This function remove the folder specified by the user.
# The input of this function is the user single spaced string with folders' names and
# the path where the oasis subjects are downloaded.
remove_folder() {
    local INPUT_STRING="$1"
    local BASE_PATH="$2"

    IFS=' ' read -r -a USER_FOLDERS <<< "$INPUT_STRING"  # Convert input string to array

    for FOLDER in "${USER_FOLDERS[@]}"; do
        echo "Deleting folder: ${BASE_PATH}/${FOLDER}"
        rm -r "${BASE_PATH}/${FOLDER}"
    done
}

# usage instructions
if [ ${#@} == 0 ]; then
    echo ""
    echo "OASIS Freesurfer download script"
    echo ""
    echo "This script downloads Freesurfer files based on a list of session ids in a csv file."
    echo "The user can manipulate the number of directories to keep after the download in order to save memory on disk."
    echo ""   
    echo "Usage: $0 input_file.csv directory_name nitrc_username scan_type"
    echo "<input_file>: A Unix formatted, comma separated file containing the following columns:"
    echo "    freesurfer_id (e.g. OAS30001_Freesurfer53_d0129)"
    echo "<directory_name>: Directory path to save Freesurfer files to"  
    echo "<nitrc_ir_username>: Your NITRC IR username used for accessing OASIS data (you will be prompted for your password)"  
    echo "<folder_to_remove>: A list of single spaced folder names to automatically eliminate at the end of the download."
    echo "   You can choose between label scripts stats surf mri."
else
    # Predefined list of valid folder in the freesurfer download
    VALID_FOLDERS=("label" "scripts" "stats" "surf" "mri") 
   	 
    # Get the input arguments
    INFILE=$1
    DIRNAME=$2
    USERNAME=$3
    UNUSED_FOLDERS=$4

    # Create the directory if it doesn't exist yet
    if [ ! -d $DIRNAME ]
    then
        mkdir $DIRNAME
    fi

    if [[ -z $UNUSED_FOLDERS ]] || validate_folders "$UNUSED_FOLDERS"; 
    then    
        echo "Script executed from: ${PWD}" 
        # Read the password of the NITRC IR account
        read -s -p "Enter your password for accessing OASIS data on NITRC IR:" PASSWORD 

        COOKIE_JAR=$(startSession)

        # Read the file
        sed 1d $INFILE | while IFS=, read -r FREESURFER_ID; do

            # Get the subject ID from the first part of the experiment ID (OAS30001 from ID OAS30001_Freesurfer53_d0129)
            SUBJECT_ID=`echo $FREESURFER_ID | cut -d_ -f1`

            # Get the days from entry from the third part of the experiment ID (d0129 from ID OAS30001_Freesurfer53_d0129)
            DAYS_FROM_ENTRY=`echo $FREESURFER_ID | cut -d_ -f3`

            # combine to form the experiment label (OAS30001_MR_d0129)
            EXPERIMENT_LABEL=${SUBJECT_ID}_MR_${DAYS_FROM_ENTRY}


            # Set project in URL based on experiment ID
            # default to OASIS3
            PROJECT_ID=OASIS3
            # If the experiment ID provided starts with OASIS4 then use project=OASIS4 in the URL
            if [[ "${EXPERIMENT_LABEL}" == "OAS4"* ]]; then
                PROJECT_ID=OASIS4
            fi

            # Get a JSESSION for authentication to XNAT
            echo "Checking for Freesurfer ID ${FREESURFER_ID} associated with ${EXPERIMENT_LABEL}."

            # Set up the download URL and make a cURL call to download the requested scans in zip format
            download_url=https://www.nitrc.org/ir/data/archive/projects/${PROJECT_ID}/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_LABEL}/assessors/${FREESURFER_ID}/files?format=zip

            download $DIRNAME/$FREESURFER_ID.zip $download_url

            # Check the zip file to make sure we downloaded something
            # If the zip file is invalid, we didn't download a scan so there is probably no scan of that type
            # If the zip file is valid, unzip and rearrange the files
            if zip -Tq $DIRNAME/$FREESURFER_ID.zip > /dev/null; then
                # We found a successfully downloaded valid zip file

                echo "Downloaded a Freesurfer (${FREESURFER_ID}) from ${EXPERIMENT_LABEL}." 

                echo "Unzipping Freesurfer and rearranging files."

                # Unzip the downloaded file
                unzip $DIRNAME/$FREESURFER_ID.zip -d $DIRNAME

                # Rearrange the files so there are fewer subfolders
                # Move the main Freesurfer subfolder up 5 levels
                # Ends up like this:
                # directory_name/OAS30001_MR_d0129/freesurfer_folders
                # directory_name/OAS30001_MR_d0129/etc
                mv $DIRNAME/$FREESURFER_ID/out/resources/DATA/files/* $DIRNAME/.

                # Change permissions on the output files
                chmod -R u=rwX,g=rwX $DIRNAME/*

                # do this so we don't have to use rm -rf. 
                rmdir $DIRNAME/$FREESURFER_ID/out/resources/DATA/files
                rmdir $DIRNAME/$FREESURFER_ID/out/resources/DATA
                rmdir $DIRNAME/$FREESURFER_ID/out/resources
                rmdir $DIRNAME/$FREESURFER_ID/out
                rmdir $DIRNAME/$FREESURFER_ID

                # Remove the Freesurfer zip file that the files were moved from
                rm -r $DIRNAME/$FREESURFER_ID.zip
                
                if [ -n "$UNUSED_FOLDERS" ]; 
                then
                    remove_folder "${UNUSED_FOLDERS[@]}" $DIRNAME/$EXPERIMENT_LABEL
                fi

            else
                echo "Could not get Freesurfer ${FREESURFER_ID} in ${EXPERIMENT_LABEL}."           
            fi

        done < $INFILE

        endSession
    fi
fi
