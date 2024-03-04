pip install transformers


Notice:If the 'transformers' library fails to import and you encounter an error, it might be due to the length of the file path exceeding the Windows default limit. To resolve this, you need to enable Long Paths in Windows 10 (Version 1607 and later) by executing the following command in PowerShell:

New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force


pip install torch

