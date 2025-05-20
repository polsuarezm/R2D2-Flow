#First add network IP in KV260 connected to HDMI:
ip addr add 192.168.42.1/24 dev eth0

#alternative -- minicom via ttyUSB1
sudo dmesg | grep ttyUSB
sudo minicom -D /dev/ttyUSB1

#check that ttyUSB connection is 115200 baud 8S1
# NO to everything
# and A -- /dev/ttyUSB1

### now connectng in remote via UART 
ssh -X root@192.168.42.1


#Prophesee interface is already installed on the SD card and a script is provided to load it and configure the video pipeline:
/usr/bin/load-prophesee-kv260-imx636.sh

#Just make sure the output ends with: “prophesee-kv-260-imx636: loaded to slot 0”. This confirms the process completed successfully.

#Power up the sensor so that its registers can be accessed:
echo on > /sys/class/video4linux/v4l-subdev3/device/power/control

#now the video pipeline should be setup
#running metavision_viewer
V4L2_HEAP=reserved V4L2_SENSOR_PATH=/dev/v4l-subdev3 metavision_viewer

yavta --capture=100 --nbufs 32 --file /dev/video0
v4l2-ctl --stream-mmap --stream-count=100 --stream-to=file.raw

#To set some camera features like ROI, biases, camera mode and so on, you can use a camera config file in JSON format. In that case, launch metavision_viewer with the option -j <camera_config_file>.json:

DISPLAY=:0.0 V4L2_HEAP=reserved V4L2_SENSOR_PATH=/dev/v4l-subdev3 metavision_viewer -j <camera_config_file>.json


