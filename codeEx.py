import math

# 거리 구하기
def dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5




# thanks to https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
# 반지름 값과 theta값을 입력 받아 중심에서 theta의 각도로 r만큼 떨어진 좌표 구하기
def rect(r, theta):
    """
    theta in degrees

    returns tuple; (float, float); (x,y)
    """

    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x, y


# thanks to https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
# x, y 좌표 값을 받아 그 좌표에 대해 중심에서 떨어진 거리, 중심에서의 각도 theta를 구하기
def polar(x, y):
    """
    returns r, theta(degrees)
    """

    r = (x ** 2 + y ** 2) ** .5
    theta = math.degrees(math.atan2(y,x))
    return r, theta


# 입력되는 각도(angle)를 파라미터와 연동하여 사용 가능한 값으로 변환(-180 ~ 180)
def angle_mod_360(angle):
    """
    Maps an angle to the interval -180, +180.

    Examples:
    angle_mod_360(362) == 2
    angle_mod_360(270) == -90

    :param angle: angle in degree
    :return: angle in degree. Between -180 and +180
    """

    n = math.floor(angle/360.0)

    angle_between_0_and_360 = angle - n*360.0

    if angle_between_0_and_360 <= 180.0:
        return angle_between_0_and_360
    else:
        return angle_between_0_and_360 - 360

    
# 차량이 시계 방향으로 움직이는지 반시계 방향으로 움직이는지 확인 후
# 반시계 방향이라면 waypoints를 반전시켜 리턴
def get_waypoints_ordered_in_driving_direction(params):
    # waypoints are always provided in counter clock wise order
    if params['is_reversed']: # driving clock wise.
        return list(reversed(params['waypoints']))
    else: # driving counter clock wise.
        return params['waypoints']


# waypoints를 받아와 그 사이사이에 factor만큼의 웨이포인트를 추가하여 리턴
def up_sample(waypoints, factor):
    """
    Adds extra waypoints in between provided waypoints

    :param waypoints:
    :param factor: integer. E.g. 3 means that the resulting list has 3 times as many points.
    :return:
    """
    p = waypoints
    n = len(p)

    return [[i / factor * p[(j+1) % n][0] + (1 - i / factor) * p[j][0],
             i / factor * p[(j+1) % n][1] + (1 - i / factor) * p[j][1]] for j in range(n) for i in range(factor)]




def get_target_point(params):
    # 위의 up_sample 함수에 get_waypoints_ordered_in_driving_direction(차량의 방향이 시계/반시계)함수의 결과값과 factor 20 대입
    waypoints = up_sample(get_waypoints_ordered_in_driving_direction(params), 20)

    # 현재 차의 위치를 car 함수에 대입
    car = [params['x'], params['y']]

    # 차와 웨이포인트들 간의 거리를 구한 후 가장 가까운 웨이포인트 인덱스 값 i_closest에 대입
    distances = [dist(p, car) for p in waypoints]
    min_dist = min(distances)
    i_closest = distances.index(min_dist)
    
    n = len(waypoints)

    # 가장 가까운 웨이포인트부터 시작되는 waypoints를 waypoints_starting_with_closest에 대입
    waypoints_starting_with_closest = [waypoints[(i+i_closest) % n] for i in range(n)]

    # 차를 중심으로 그려질 원의 반지름
    r = params['track_width'] * 0.9

    # waypoints_starting_with_closest 배열에서 해당 index의 waypoint가 차 중심의 원 바깥에 존재 시 False가 들어가있는 배열
    # False가 가장 처음 나온 index값을 i_first_outside에 넣음
    # 결론 : 원 바깥에 있는 waypoint 중 가장 가까운 waypoint의 index값 반환
    is_inside = [dist(p, car) < r for p in waypoints_starting_with_closest]
    i_first_outside = is_inside.index(False)

    # 원의 크기가 전체 트랙만큼 클 경우엔 가장 가까운 waypoints반환
    if i_first_outside < 0:  # this can only happen if we choose r as big as the entire track
        return waypoints[i_closest]

    # 위에서 구한 값들로 차 중심의 원 바깥의 가장 가까운 waypoint 값 반환
    return waypoints_starting_with_closest[i_first_outside]


def get_target_steering_degree(params):
    # 위의 get_target_point로 가야할 waypoint = x,y 좌표를 받아와 차와의 거리 구함
    # 차의 현재 각도 가져오기
    tx, ty = get_target_point(params)
    car_x = params['x']
    car_y = params['y']
    dx = tx-car_x
    dy = ty-car_y
    heading = params['heading']

    # 차의 위치가 0, 0일 때(dx, dy)로 가기 위한 방향
    _, target_angle = polar(dx, dy)

    # 차가 목표 위치로 가기위한 바퀴의 각도
    steering_angle = target_angle - heading

    return angle_mod_360(steering_angle)


# 현재 차량의 바퀴 방향과 타겟으로 가기 위한 바퀴 방향의 차이가 적을 수록 보상을 많이 주도록 설정
# 두 차이가 60도 이상일 경우 0.01점 반환
def score_steer_to_point_ahead(params):
    best_stearing_angle = get_target_steering_degree(params)
    steering_angle = params['steering_angle']
    
    error = (steering_angle - best_stearing_angle) / 60.0  # 60 degree is already really bad

    score = 1.0 - abs(error)

    return max(score, 0.01)  # optimizer is rumored to struggle with negative numbers and numbers too close to zero

# 메인 함수
# 위의 함수에서 구해진 score값 리턴
def reward_function(params):
    return float(score_steer_to_point_ahead(params))


def get_test_params():
    return {
        'x': 0.7,
        'y': 1.05,
        'heading': 160.0,
        'track_width': 0.45,
        'is_reversed': False,
        'steering_angle': 0.0,
        'waypoints': [
            [0.75, -0.7],
            [1.0, 0.0],
            [0.7, 0.52],
            [0.58, 0.7],
            [0.48, 0.8],
            [0.15, 0.95],
            [-0.1, 1.0],
            [-0.7, 0.75],
            [-0.9, 0.25],
            [-0.9, -0.55],
        ]
    }


def test_reward():
    params = get_test_params()

    reward = reward_function(params)

    print("test_reward: {}".format(reward))

    assert reward > 0.0


def test_get_target_point():
    result = get_target_point(get_test_params())
    expected = [0.33, 0.86]
    eps = 0.1

    print("get_target_point: x={}, y={}".format(result[0], result[1]))

    assert dist(result, expected) < eps


def test_get_target_steering():
    result = get_target_steering_degree(get_test_params())
    expected = 46
    eps = 1.0

    print("get_target_steering={}".format(result))

    assert abs(result - expected) < eps


def test_angle_mod_360():
    eps = 0.001

    assert abs(-90 - angle_mod_360(270.0)) < eps
    assert abs(-179 - angle_mod_360(181)) < eps
    assert abs(0.01 - angle_mod_360(360.01)) < eps
    assert abs(5 - angle_mod_360(365.0)) < eps
    assert abs(-2 - angle_mod_360(-722)) < eps

def test_upsample():
    params = get_test_params()
    print(repr(up_sample(params['waypoints'], 2)))

def test_score_steer_to_point_ahead():
    params_l_45 = {**get_test_params(), 'steering_angle': +45}
    params_l_15 = {**get_test_params(), 'steering_angle': +15}
    params_0 = {**get_test_params(), 'steering_angle': 0.0}
    params_r_15 = {**get_test_params(), 'steering_angle': -15}
    params_r_45 = {**get_test_params(), 'steering_angle': -45}

    sc = score_steer_to_point_ahead

    # 0.828, 0.328, 0.078, 0.01, 0.01
    print("Scores: {}, {}, {}, {}, {}".format(sc(params_l_45), sc(params_l_15), sc(params_0), sc(params_r_15), sc(params_r_45)))


def run_tests():
    test_angle_mod_360()
    test_reward()
    test_upsample()
    test_get_target_point()
    test_get_target_steering()
    test_score_steer_to_point_ahead()

    print("All tests successful")


# run_tests()
