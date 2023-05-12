import Box2D.b2
import pygame
import math
from random import randint, uniform, shuffle
from itertools import combinations

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

world = Box2D.b2World(gravity=(0, 100))

class Floor():
    def __init__(self, world):
        self.ground_body = world.CreateStaticBody(position=(0, 400))
        self.ground_body.CreatePolygonFixture(box=(650, 10))

class Shape():
    def __init__(self, world):
        self.box = world.CreateDynamicBody(position=(100, 300))

        num_sides = randint(3, 8)
        distance = 30 * math.sin(math.pi / num_sides)
        self.vertices = [(distance * math.cos(2 * math.pi / num_sides * i) + uniform(-10, 10),
                    distance * math.sin(2 * math.pi / num_sides * i) + uniform(-10, 10))
                    for i in range(num_sides)]
        self.box_shape = Box2D.b2PolygonShape(vertices=self.vertices)
        self.box_fixture = self.box.CreateFixture(shape=self.box_shape, density=1, friction=0.3)

class Joint():
    def __init__(self, world, shape1, shape2):
        joint_def = Box2D.b2RevoluteJointDef(
            bodyA=shape1.box,
            bodyB=shape2.box,
            localAnchorA=shape1.vertices[randint(0, len(shape1.vertices)-1)],
            localAnchorB=shape2.vertices[randint(0, len(shape2.vertices)-1)],
        )

        num_sides = randint(3, 8)
        distance = 30 * math.sin(math.pi / num_sides)
        print(distance)

        joint_def.enableMotor = True  # Enable the motor
        joint_def.motorSpeed = 2 * 10 ** 3 # Initial motor speed (change as needed)
        joint_def.maxMotorTorque = 1000 * 10 ** 4 # Maximum torque the motor can exert (change as needed)

        self.joint = world.CreateJoint(joint_def)
        self.motorSpeed = 2 * 10 ** 3


class Creature():
    def __init__(self, world):
        self.n_shapes = 5
        num_chains = 2
        chain_lengths = [3, 2]  # specify the lengths of each chain
        
        self.shapes = []
        for i in range(self.n_shapes):
            self.shapes.append(Shape(world))

        connections = []
        self.joints = []

        # connect shapes within each chain
        for i in range(num_chains):
            chain_start = sum(chain_lengths[:i])  # index of first shape in chain
            for j in range(chain_lengths[i]-1):
                connections.append((self.shapes[chain_start+j], self.shapes[chain_start+j+1]))

        # connect chains together
        for i in range(num_chains-1):
            chain1_start = sum(chain_lengths[:i])
            chain2_start = sum(chain_lengths[:i+1])
            connections.append((self.shapes[chain1_start], self.shapes[chain2_start]))

        shuffle(connections)

        for i in range(len(connections)):
            self.joints.append(Joint(world, connections[i][0], connections[i][1]))
        

creature = Creature(world)
ground_body = Floor(world)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    world.Step(1.0/60.0, 6, 2)
    screen.fill((255, 255, 255))

    for fixture in ground_body.ground_body.fixtures:
        shape = fixture.shape
        vertices = [ground_body.ground_body.transform * v for v in shape.vertices]
        pygame.draw.polygon(screen, (128, 128, 128), vertices)
    
    index = 0
    for shape in creature.shapes:
        index += 1
        for fixture in shape.box.fixtures:
            sub_shape = fixture.shape
            vertices = [shape.box.transform * v for v in sub_shape.vertices]
            pygame.draw.polygon(screen, (30 * index, 90, 90), vertices)

    for joint in creature.joints:
        joint.joint.motorSpeed = randint(-10, 10) * 20**3

    keys = pygame.key.get_pressed()
    num_sides = randint(3, 8)
    distance = 30 * math.sin(math.pi / num_sides)
    force = Box2D.b2Vec2(0, 0)
    if keys[pygame.K_LEFT]:
        force += (-500 * distance**3, 0)
    if keys[pygame.K_RIGHT]:
        force += (500 * distance**3, 0)
    if keys[pygame.K_UP]:
        force += (0, -500 * distance**3)
    if keys[pygame.K_DOWN]:
        force += (0, 500 * distance**3)
    creature.shapes[0].box.ApplyForceToCenter(force, True)

    pygame.display.flip()
    clock.tick(60)
