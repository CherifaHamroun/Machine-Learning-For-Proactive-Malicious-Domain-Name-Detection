# Machine learning for proactive malicious domain name detection

### Install requirements.txt 
```
pip3 install -r requirements.txt
```

### Nostril installation 
nostril :
```
git clone https://github.com/casics/nostril.git
cd nostril
sudo python3 -m pip install .
```

### Crontab configuration 
phish_getter.sh (at t0)
spams_getter.sh (at t0)
trends_getter.sh (at t0)
reprocessor.sh (at t1 = t0+10min)
retrainer.sh (at t1+2h)