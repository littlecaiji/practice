#!python3
import os
os.system("cp -r ~/.kde/share/apps/okular/docdata ./")
os.system("git add *")
os.system("git commit -m \"Daily Update\"")
os.system("git push origin master")
