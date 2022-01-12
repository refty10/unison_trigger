from concurrent.futures import ThreadPoolExecutor
import subprocess
import time


class MonitorCtl:
    def __init__(self):
        self.status = True
        self.brightness = self.get_brightness()

    def get_brightness(self):
        res = subprocess.run("brightness -l", shell=True, capture_output=True)
        info = res.stdout.decode('utf-8')
        return float(info.split()[-1])

    def set_brightness(self, value):
        subprocess.run(f'brightness {value}', shell=True)

    def sleep(self):
        if not self.status:
            return
        self.brightness = self.get_brightness()
        self.set_brightness(0)
        self.status = False

    def wake_up(self):
        if self.status:
            return
        self.set_brightness(self.brightness)
        self.status = True


def main():
    mc = MonitorCtl()
    mc.sleep()
    time.sleep(3)
    mc.wake_up()


if __name__ == '__main__':
    main()
