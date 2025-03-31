import spidev

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 50000

#send onebyte, read one byte
reponse = spi.xfer2([0x00])
print("SPI response: ", response)

spi.close()
