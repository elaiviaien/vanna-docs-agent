#!/usr/bin/env sh
# fix perms on our host-mounted dirs
chown -R appuser:appuser /app/evaluation_data /app/logs

# now run the passed-in command as appuser
exec gosu appuser "$@"
