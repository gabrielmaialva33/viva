#!/bin/bash
export PATH=$PATH:/usr/local/bin:/opt/homebrew/bin
mix deps.get
mix credo --strict --format=json > credo_results.json 2> credo_error.log
ls -l credo_results.json
cat credo_error.log
