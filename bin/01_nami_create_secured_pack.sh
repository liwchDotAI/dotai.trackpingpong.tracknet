# Build secured pack, entry Python script must be provided
ENTRY_SCRIPT=${1}
SCRIPT_NAME=`basename ${ENTRY_SCRIPT%.*}`
pyarmor pack -O secured_pack_for_${SCRIPT_NAME} ${ENTRY_SCRIPT}
echo Secured exectuable pack in secured_pack_for_${SCRIPT_NAME}/ folder, your can run secured_pack_for_${SCRIPT_NAME}/${SCRIPT_NAME}/${SCRIPT_NAME} to try.
