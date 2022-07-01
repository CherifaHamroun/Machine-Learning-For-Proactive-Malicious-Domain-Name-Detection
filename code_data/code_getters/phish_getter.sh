#!/bin/bash
pushd ./code_implementation/code_data/data_sources
wget -q -O - "https://github.com/mitchellkrogza/Phishing.Database/blob/master/ALL-phishing-domains.tar.gz?raw=true" | tar xz 
cat ALL-phishing-domains.txt all_phishing.txt | sort | uniq > all_phishing.txt
rm ALL-phishing-domains.txt
popd