#!/bin/bash
echo "Download gloves and unzip them..."

curl -H "Authorization: Bearer ya29.GltIBBxLQoj8_lSGNSTszXnHmpvW1V2UkpVpM425teDOYo4HoAaVMfshn3U_rBIgDDnQPv_Ryhe5D0PNBRGmVmdOda6g0VbSoerCiu_XtgmwoPS4uffiS_w-9ciK" https://www.googleapis.com/drive/v3/files/0B2hNk0_VowQmSUNwWEpSQ1VXNTA?alt=media -o gloves.zip
unzip -u-o gloves.zip
rm -rf gloves.zip

echo "Finish downloading gloves!!"