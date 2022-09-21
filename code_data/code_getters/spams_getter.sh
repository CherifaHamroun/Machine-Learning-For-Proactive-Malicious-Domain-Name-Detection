#!/bin/bash
pushd ./code_implementation/code_data/data_sources
wget -q -O - "https://raw.githubusercontent.com/gsocgsoc/spam-domains/master/spamdomains.txt">>spamdomains1.txt
wget -q -O - "https://raw.githubusercontent.com/no-cmyk/Search-Engine-Spam-Blocklist/master/blocklist.txt">>spamdomains2.txt
wget -q -O - "https://raw.githubusercontent.com/ThioJoe/YT-Spam-Lists/main/SpamDomainsList.txt" >>spamdomains3.txt
wget -q -O - "https://raw.githubusercontent.com/groundcat/disposable-email-domain-list/master/domains.txt" >>spamdomains4.txt
wget -q -O - "https://raw.githubusercontent.com/zaosoula/email-spam-domains/master/domains.txt" >>spamdomains5.txt
wget -q -O - "https://raw.githubusercontent.com/tsirolnik/spam-domains-list/master/spamdomains.txt" >>spamdomains6.txt
cat spamdomains*.txt > spamdomainsall.txt 
sort spamdomainsall.txt > spamdomainsall.txt
diff spamdomainsall.txt all_spams.txt | grep "<" | sed 's/^< //g' > diff_spam.txt
cat spamdomainsall.txt all_spams.txt | sort | uniq > all_spams.txt
wait
sed -i 's/#.*$//' all_spams.txt
sed -i  '/^[[:space:]]*$/d' all_spams.txt
rm spamdomains*.txt
popd
