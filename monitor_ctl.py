import subprocess

class MonitorCtl:
  @classmethod
  def sleep():
    subprocess.run("pmset displaysleepnow", shell=True)
    
  @classmethod
  def wake_up():
    subprocess.run("caffeinate -u -t 1", shell=True)