#encoding: utf-8
from __future__ import division
from nodebox.graphics import *
import pymunk
import pymunk.pyglet_util
import random, math
import numpy as np

space = pymunk.Space()

def createBody(x,y,shape,*shapeArgs):
    body = pymunk.Body()
    body.position = x, y
    s = shape(body, *shapeArgs)
    s.mass = 1
    s.friction = 1
    space.add(body, s)
    return s #shape!!!

s0=createBody(300, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s0.score=0
s3=createBody(200, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s3.color = (0, 255, 0, 255)
s3.score=0
s3.body.Q=[[0, 0], [0, 0], [0, 0]]
#s3.body.Q=[[0, 0], [1, -1], [-1, 1]]
#Q=[нічого[залишати, змінювати], об'єкт[залишати, змінювати], антиоб'єкт[залишати, змінювати]]
s3.body.action=0 # 0 - залишати, 1 - змінювати (випадковий кут)
s1=createBody(300, 200, pymunk.Circle, 10, (0,0))
S2=[]
for i in range(1):
    s2=createBody(350, 250, pymunk.Circle, 10, (0,0))
    s2.color = (255, 0, 0, 255)
    S2.append(s2)


def getAngle(x,y,x1,y1):
    return math.atan2(y1-y, x1-x)

def getDist(x,y,x1,y1):
    return ((x-x1)**2+(y-y1)**2)**0.5

def inCircle(x,y,cx,cy,R):
    if (x-cx)**2+(y-cy)**2 < R**2:
        return True
    return False

def inSector(x,y,cx,cy,R,a):
    angle=getAngle(cx,cy,x,y)
    a=a%(2*math.pi)
    angle=angle%(2*math.pi)
    if inCircle(x,y,cx,cy,R) and a-0.5<angle<a+0.5:
        return True
    return False

def strategy(b=s3.body):
    v=100
    a=b.angle
    b.velocity=v*cos(a), v*sin(a)
    x,y=b.position
    R=getDist(x,y,350,250)
    #print R, b.angle
    line(x,y,*s1.body.position,stroke=Color(0))
    if canvas.frame%100==0:
        if R>180:
            b.angle=getAngle(x,y,350,250)
        else:
            b.angle=getAngle(x,y,*s1.body.position) #2*math.pi*random.random()

def strategy2(b=s3.body):
    u"""Удосконалена стратегія робота, який сканує сектор ультразвуковим сенсором з Q-learning."""
    v = 100  # Швидкість робота
    a = b.angle  # Кут нахилу робота
    b.velocity = (v * cos(a), v * sin(a))  # Вектор швидкості на основі кута
    x, y = b.position  # Поточна позиція робота
    R = getDist(x, y, 350, 250)  # Відстань до центру (точка (350, 250))
    ellipse(x, y, 200, 200, stroke=Color(0.5))  # Візуалізація робота (еліпс)

    # Візуалізація напрямку робота
    line(x, y, x + 100 * cos(a + 0.5), y + 100 * sin(a + 0.5), stroke=Color(0.5))  # Лінія під кутом +0.5
    line(x, y, x + 100 * cos(a - 0.5), y + 100 * sin(a - 0.5), stroke=Color(0.5))  # Лінія під кутом -0.5

    if canvas.frame % 10 == 0:  # Кожні 10 кадрів
        # Перевіряємо наявність об'єктів s1 та s2 в секторі
        inS = inSector(s1.body.position[0], s1.body.position[1], x, y, 100, a)  # Об'єкт s1
        inS2 = inSector(S2[0].body.position[0], S2[0].body.position[1], x, y, 100, a)  # Антиоб'єкт s2

        # Установлюємо стан та винагороду на основі наявності об'єктів
        if inS:
            state = 1  # Стан, якщо знайдено s1
            reward = 1 if b.action == 0 else -1  # Винагорода за правильну дію (0 - залишити)
        elif inS2:
            state = 2  # Стан, якщо знайдено s2
            reward = -1 if b.action == 0 else 1  # Винагорода за неправильну дію
        else:
            state = 0  # Нічого не знайдено
            reward = 0  # Винагорода відсутня

        # Застосування знижуваного навчання
        alpha = 0.1  # Коефіцієнт навчання (визначає, як швидко модель адаптується)
        gamma = 0.9  # Дисконтуючий фактор для винагороди (оцінка майбутніх винагород)
        # Оновлення Q-таблиці з використанням формули навчання
        b.Q[state][b.action] += alpha * (reward + gamma * np.max(b.Q[state]) - b.Q[state][b.action])

        # Вибір дії з епсилон-жадібною стратегією
        epsilon = 0.1  # Ймовірність вибору випадкової дії (експериментування)
        if random.random() < epsilon:
            b.action = random.choice([0, 1])  # Випадкова дія (залишити чи змінити)
        else:
            b.action = np.argmax(b.Q[state])  # Вибір найкращої дії на основі Q-таблиці

        # Змінюємо кут, якщо потрібно
        if b.action:  # Якщо обрано змінити кут
            b.angle = 2 * math.pi * random.random()  # Випадковий новий кут

        # Запобігання виїзду робота за межі
        if R > 180:  # Якщо відстань до центру більше 180
            b.angle = getAngle(x, y, 350, 250)  # Повертаємо робота до центру


def scr(s,s0,s3,p=1):
    bx,by=s.body.position
    s0x,s0y=s0.body.position
    s3x,s3y=s3.body.position
    if not inCircle(bx,by,350,250,180):
        if getDist(bx,by,s0x,s0y)<getDist(bx,by,s3x,s3y):
            s0.score=s0.score+p
        else:
            s3.score=s3.score+p
        s.body.position=random.randint(200,400),random.randint(200,300)

def score():
    u"""визначає переможця"""
    scr(s1,s0,s3)
    for s in S2:
        scr(s,s0,s3,p=-1)


def manualControl():
    u"""Керування роботом з мишки або клавіатури"""
    v=10 # швидкість
    b=s0.body
    a=b.angle
    x,y=b.position
    vx,vy=b.velocity
    if canvas.keys.char=="a":
        b.angle-=0.1
    if canvas.keys.char=="d":
        b.angle+=0.1
    if canvas.keys.char=="w":
        b.velocity=vx+v*cos(a), vy+v*sin(a)
    if canvas.mouse.button==LEFT:
        b.angle=getAngle(x,y,*canvas.mouse.xy)
        b.velocity=vx+v*cos(a), vy+v*sin(a)

def simFriction():
    for s in [s0,s1,s3]+S2:
        s.body.velocity=s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
        s.body.angular_velocity=s.body.angular_velocity*0.9
        #s.body.update_velocity(s.body, (0.,0.), 0.9, 0.02)

draw_options = pymunk.pyglet_util.DrawOptions()

def draw(canvas):
    canvas.clear()
    fill(0,0,0,1)
    text("%i %i"%(s0.score,s3.score),20,20)
    nofill()
    ellipse(350, 250, 350, 350, stroke=Color(0))
    manualControl()
    strategy2()
    score()
    simFriction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)
