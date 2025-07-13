import motor
import motor_pair
import runloop
from hub import motion_sensor, port
import utime
import color_sensor

def getGyro():
    return -1*motion_sensor.tilt_angles()[0]

def printGyro():
    print(getGyro())

#def raspi(c):
#    return input(f"{c}:")

def adamGibiDon(angle:int, turnDir="f"):
    init_gyro = getGyro()
    mot1 = None
    mot2 = None
    if turnDir == "f" or turnDir=="F":
        mot1=port.F # sol motor
        mot2=port.B # sag motor
    else:
        mot1=port.B # sag motor
        mot2=port.F # sol motor

    oran_ti1 = -1*int((angle-init_gyro) * 8/11)
    oran_ti2 = -1*int((angle-init_gyro) * 6/11)
    oran_ti3 = -1*int(4/11 * (angle-init_gyro))

    if angle>init_gyro: #angle>0
        while angle >= getGyro():
            if 1/7 * angle > getGyro():
                motor.run(mot1, oran_ti3)
            elif 2/7 * angle > getGyro():
                motor.run(mot1, oran_ti1)
            elif 4/7 * angle > getGyro():
                motor.run(mot1, oran_ti2)
            else:
                motor.run(mot1, -280)

        motor.stop(mot1, stop=motor.HOLD)
    else:
        while angle <= getGyro():#angle+init_gyro
            if 1/7 * angle < getGyro():
                motor.run(mot2, oran_ti3)
            elif 2/7 * angle < getGyro():
                motor.run(mot2, oran_ti1)
            elif 4/7 * angle < getGyro():
                motor.run(mot2, oran_ti2)
            else:
                motor.run(mot2, 280)

    motor_pair.stop(motor_pair.PAIR_1, stop=motor.HOLD)
    utime.sleep_ms(40)

def moveWithGyro(velocity:int, angle:int, duration:int, ST = motor.HOLD):
    utime.sleep_ms(40)
    motor_pair.stop(motor_pair.PAIR_1, stop=ST)
    start_time = utime.ticks_ms()
    while utime.ticks_diff(utime.ticks_ms(), start_time) < duration:
        adj = int(angle-getGyro())
        motor_pair.move_tank(motor_pair.PAIR_1, velocity + adj, velocity - adj)


    motor_pair.stop(motor_pair.PAIR_1,stop=ST)
    utime.sleep_ms(30)

def twoWheelTurn(angle:int):
    motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)
    utime.sleep_ms(100)
    init_gyro = getGyro()
    oran_ti1 = -1*int((angle-init_gyro) * 4/11)
    oran_ti2 = -1*int((angle-init_gyro) * 3/11)
    oran_ti3 = -1*int(2/11 * (angle-init_gyro))
    if angle>init_gyro:
        while angle>= getGyro():
            if 1/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 4/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti1, oran_ti1)
            elif 5/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti2, oran_ti2)
            else:
                motor_pair.move_tank(motor_pair.PAIR_1, 140, -1*140)

        motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)
        return
    else:
        while angle <= getGyro():
            if 1/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 4/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 5/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            else:
                motor_pair.move_tank(motor_pair.PAIR_1, -1*140, 140)

        motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)
        return

def gyroRun(velocity:int, angle:int):
    adj = int(angle-getGyro())
    motor_pair.move_tank(motor_pair.PAIR_1, velocity + adj, velocity - adj)

def fastfastTurn(angle:int, turnDir="f"):
    # motor.run(port.A, 1000) positive
    motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)
    utime.sleep_ms(30)
    init_gyro = getGyro()
    if turnDir == "f" or turnDir=="F":
        mot1=port.F # sol motor
        mot2=port.B # sag motor
    else:
        mot1=port.B # sag motor
        mot2=port.F # sol motor
     #angle<init_gyro
    oran_ti1 = -1*int((angle-init_gyro) * 9/10)
    oran_ti2 = -1*int((angle-init_gyro) * 7/10)
    oran_ti3 = -1*int(5/10 * (angle-init_gyro))
    
    if angle>init_gyro: #angle>0
        while angle >= getGyro():
            if 1/7 * angle > getGyro():
                motor.run(mot1, oran_ti3)
            elif 4/7 * angle > getGyro():
                motor.run(mot1, oran_ti1)
            elif 5/7 * angle > getGyro():
                motor.run(mot1, oran_ti2)
            else:
                motor.run(mot1, -350)

        motor.stop(mot1, stop=motor.HOLD)
    else:
        while angle <= getGyro():#angle+init_gyro
            if 1/7 * angle < getGyro():
                motor.run(mot2, oran_ti3)
            elif 4/7 * angle < getGyro():
                motor.run(mot2, oran_ti1)
            elif 5/7 * angle < getGyro():
                motor.run(mot2, oran_ti2)
            else:
                motor.run(mot2, 350)

        motor.stop(mot2, stop=motor.HOLD)

    motor_pair.stop(motor_pair.PAIR_1, stop=motor.HOLD)
    utime.sleep_ms(80)

def twoWheelSlow(angle):
    motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)
    utime.sleep_ms(100)
    init_gyro = getGyro()
    oran_ti1 = -1*int((angle-init_gyro) * 6/22)
    oran_ti2 = -1*int((angle-init_gyro) * 4/22)
    oran_ti3 = -1*int(3/22 * (angle-init_gyro))
    if angle>init_gyro:
        while angle>= getGyro():
            if 1/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 4/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti1, oran_ti1)
            elif 5/7 * angle > getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti2, oran_ti2)
            else:
                motor_pair.move_tank(motor_pair.PAIR_1, 140, -1*140)

        motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)

    else:
        while angle <= getGyro():
            if 1/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 4/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            elif 5/7 * angle < getGyro():
                motor_pair.move_tank(motor_pair.PAIR_1, -1*oran_ti3, oran_ti3)
            else:
                motor_pair.move_tank(motor_pair.PAIR_1, -1*140, 140)

        motor_pair.stop(motor_pair.PAIR_1,stop=motor.HOLD)

    utime.sleep_ms(50)

def gyroTurn(angle:int, spid="f",turnDir="f", ST=motor.HOLD):
    # motor.run(port.A, 1000) positive
    #motor_pair.stop(motor_pair.PAIR_1,stop=ST)
    utime.sleep_ms(30)
    init_gyro = getGyro()
    if turnDir == "f" or turnDir=="F":
        mot1=port.F # sol motor
        mot2=port.B # sag motor
    else:
        mot1=port.B # sag motor
        mot2=port.F # sol motor
    if spid=="s" or spid=="S":
        #angle<init_gyro
        oran_ti1 = -1*int((angle-init_gyro)* 8/11)
        oran_ti2 = -1*int((angle-init_gyro) * 6/11)
        oran_ti3 = -1*int(4/11 * (angle-init_gyro))
        
    else:
        #angle<init_gyro
        oran_ti1 = -1*int((angle-init_gyro) * 4/5)
        oran_ti2 = -1*int((angle-init_gyro) * 3/5)
        oran_ti3 = -1*int(2/5 * (angle-init_gyro))
    if angle>init_gyro: #angle>0
        while angle >= getGyro(): # angle+init_gyro
            if 1/7 * angle > getGyro():
                motor.run(mot1, oran_ti3)
            elif 4/7 * angle > getGyro():
                motor.run(mot1, oran_ti1)
            elif 5/7 * angle > getGyro():
                motor.run(mot1, oran_ti2)
            else:
                motor.run(mot1, -280)

        motor.stop(mot1, stop=motor.HOLD)
    else:
        while angle <= getGyro():

            if 1/7 * angle < getGyro():
                motor.run(mot2, oran_ti3)
            elif 4/7 * angle < getGyro():
                motor.run(mot2, oran_ti1)
            elif 5/7 * angle < getGyro():
                motor.run(mot2, oran_ti2)
            else:
                motor.run(mot2, 280)

        motor.stop(mot2, stop=motor.HOLD)
    motor_pair.stop(motor_pair.PAIR_1, stop=motor.HOLD)
    utime.sleep_ms(140)

def checkColor(minimum:int, maximum:int):
    W = (color_sensor.rgbi(port.A)[0] + color_sensor.rgbi(port.A)[1] + color_sensor.rgbi(port.A)[2])/3
    if W > minimum and W < maximum:
        return True
    else:
        return False


