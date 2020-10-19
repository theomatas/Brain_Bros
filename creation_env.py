import os
f=open('creation_env.txt',"r",encoding="utf-8")
commandes=f.read().split('\n')
f.close()
for commande in commandes :
	os.system(commande)