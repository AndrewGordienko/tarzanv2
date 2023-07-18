import Box2D.b2
# import pygame
import math
from random import randint, uniform, shuffle
from itertools import combinations
from brain import Agent
import torch

agent = Agent()
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
# pygame.init()
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# clock = pygame.time.Clock()

world = Box2D.b2World(gravity=(0, 100))

class Floor():
    def __init__(self, world):
        self.ground_body = world.CreateStaticBody(position=(0, 400))
        self.ground_body.CreatePolygonFixture(box=(650, 10))

class Shape():
    def __init__(self, world):
        self.box = world.CreateDynamicBody(position=(325, 300))

        num_sides = randint(4, 8)
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

class RayCastCallback(Box2D.b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.hit = False
        self.distance = -1.0  # Initialize distance to a negative value

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.distance = fraction  # Store the fraction value as the distance
        return 0  # Returning 0 terminates the raycast after the first hit

class Simulation():
    def run_through(self, direction_moving):
        creature = Creature(world)
        ground_body = Floor(world)

        self.observation = [0] * 17
        self.action = 0
        self.reward = 0
        self.done = False
        self.score = 0
        self.step = 0

        self.reference_position = Box2D.b2Vec2(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 10)
        self.prev_distance = 0

        while not self.done:
            world.Step(1.0/60.0, 6, 2)
            self.step += 1
            
            if self.step == 1000:
                self.done = True

            for fixture in ground_body.ground_body.fixtures:
                shape = fixture.shape
                vertices = [ground_body.ground_body.transform * v for v in shape.vertices]
            
            index = 0
            touch_inputs = [0] * 5
            raycast_result = [-1.0] * 8  # List to store the distances until contact for the eight directions
            directions = [0, 1, 2, 3, 4, 5, 6, 7]  # Directions in multiples of 45 degrees

            for shape in creature.shapes:
                index += 1
                for fixture in shape.box.fixtures:
                    sub_shape = fixture.shape
                    vertices = [shape.box.transform * v for v in sub_shape.vertices]

                    for contact_edge in shape.box.contacts:
                        contact = contact_edge.contact
                        if contact.fixtureA.body == shape.box and contact.fixtureB.body == ground_body.ground_body:
                            touch_inputs[index-1] = 1
                            break

                    if index == 3:  # For the middle shape (shape number three)
                        center = shape.box.transform * Box2D.b2Vec2(0, 0)  # Center point of the shape
                        raycast_length = 100  # Length of the rays

                        callback = RayCastCallback()  # Create an instance of the callback object

                        for direction in directions:
                            angle = direction * (math.pi / 4)  # Convert the direction to radians
                            raycast_end = center + raycast_length * Box2D.b2Vec2(math.cos(angle), math.sin(angle))

                            world.RayCast(callback, center, raycast_end)  # Perform the raycast for each direction
                            if callback.hit:
                                raycast_result[direction] = raycast_length * callback.distance

                            callback.hit = False  # Reset the hit flag for the next raycast

            joint_angles = [joint.joint.angle for joint in creature.joints]
            self.observation = raycast_result + touch_inputs + joint_angles

            speeds = agent.choose_action(self.observation, direction_moving)
            joint_index = 0
            for joint in creature.joints:
                joint.joint.motorSpeed = speeds[joint_index].item()
                joint_index += 1

            # Calculate the distance from the reference position (middle of the floor)
            current_position = creature.shapes[2].box.position
            distance_from_reference = (current_position - self.reference_position).length

            # Check if the middle body touches the ground
            middle_shape = creature.shapes[2]
            middle_body = middle_shape.box
            for contact_edge in middle_body.contacts:
                contact = contact_edge.contact
                if contact.fixtureA.body == middle_body and contact.fixtureB.body == ground_body.ground_body:
                    self.done = True
                    self.score -= 100
                    break

            # Check if the distance from the reference increased or decreased
            if distance_from_reference > self.prev_distance:
                self.reward = 1  # Set reward to 1 if the robot moves away from the reference position
                self.score += 1
            elif distance_from_reference < self.prev_distance:
                self.reward = -1  # Set reward to -1 if the robot moves towards the reference position
                self.score += -1
            else:
                self.reward = 0  # Set reward to 0 if there is no change in distance
            
            index = 0
            touch_inputs = [0] * 5
            raycast_result = [-1.0] * 8  # List to store the distances until contact for the eight directions
            directions = [0, 1, 2, 3, 4, 5, 6, 7]  # Directions in multiples of 45 degrees

            for shape in creature.shapes:
                index += 1
                for fixture in shape.box.fixtures:
                    sub_shape = fixture.shape
                    vertices = [shape.box.transform * v for v in sub_shape.vertices]

                    for contact_edge in shape.box.contacts:
                        contact = contact_edge.contact
                        if contact.fixtureA.body == shape.box and contact.fixtureB.body == ground_body.ground_body:
                            touch_inputs[index-1] = 1
                            break

                    if index == 3:  # For the middle shape (shape number three)
                        center = shape.box.transform * Box2D.b2Vec2(0, 0)  # Center point of the shape
                        raycast_length = 100  # Length of the rays

                        callback = RayCastCallback()  # Create an instance of the callback object

                        for direction in directions:
                            angle = direction * (math.pi / 4)  # Convert the direction to radians
                            raycast_end = center + raycast_length * Box2D.b2Vec2(math.cos(angle), math.sin(angle))

                            world.RayCast(callback, center, raycast_end)  # Perform the raycast for each direction
                            if callback.hit:
                                raycast_result[direction] = raycast_length * callback.distance

                            callback.hit = False  # Reset the hit flag for the next raycast

            joint_angles = [joint.joint.angle for joint in creature.joints]
            self.observation_ = raycast_result + touch_inputs + joint_angles

            self.observation = list(self.observation)
            self.observation += [direction_moving] * 17
            self.observation = torch.Tensor(self.observation)

            self.observation_ = list(self.observation_)
            self.observation_ += [direction_moving] * 17
            self.observation_ = torch.Tensor(self.observation_)

            agent.memory.add(self.observation, speeds.detach().numpy(), self.reward, self.observation_, self.done)

            # Update the previous distance
            self.prev_distance = distance_from_reference

          
