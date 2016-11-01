# Push artifacts up
Push-AppveyorArtifact appveyor/SAM.mdf
Push-AppveyorArtifact appveyor/SAM_log.ldf

# Use mdf/ldf to create SAM db
$mdfFile = "c:\projects\healthcareai-py\appveyor\SAM.mdf"
$ldfFile = "c:\projects\healthcareai-py\appveyor\SAM_log.ldf"

sqlcmd -b -S "(local)\SQL2012SP1" -Q "CREATE DATABASE [SAM] ON (FILENAME = '$mdfFile'), (FILENAME = '$ldfFile') for ATTACH"