# Build secured source distribution
ENTRY_SCRIPT=${1}
SCRIPT_NAME=`basename ${ENTRY_SCRIPT%.*}`
pyarmor obfuscate -O secured_src_for_${SCRIPT_NAME} ${ENTRY_SCRIPT}
echo Secured source codes in secured_src_for_${SCRIPT_NAME}/ folder.
