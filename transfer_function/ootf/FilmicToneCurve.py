import numpy as np

class CurveParamsUser:
    def __init__(self) -> None:
        self.toe_strength = 0.0
        self.toe_length = 0.5
        self.shoulder_strength = 0.0
        self.shoulder_length = 0.5

        self.shoulder_angle = 0
        self.gamma = 1

class CurveParamsDirect:
    def __init__(self) -> None:
        self.Reset()

    def Reset(self):
        self.x0 = 0.25
        self.y0 = 0.25
        self.x1 = 0.75
        self.y1 = 0.75
        self.W = 1
        self.gamma = 1
        self.overshoot_x = 0
        self.overshoot_y = 0


class CurveSegment:
    def __init__(self) -> None:
        self.Reset()

    def Reset(self):
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.lnA = 0.0
        self.B = 1.0

    def Eval(self, x):
        x0 = (x - self.offset_x) * self.scale_x

        y0 = np.piecewise(x0, [x0 > 0], [
            lambda x0: np.exp(self.lnA + self.B * np.log(x0)),
            lambda x0: 0
        ])
        # if x0 > 0:
        #     y0 = np.exp(self.lnA + self.B * np.log(x0))

        return y0 * self.scale_y + self.offset_y

# mid = CurveSegment() 
# toe = CurveSegment()
# shoulder = CurveSegment()

def solveAB(x0, y0, m):
    B = (m*x0)/y0
    lnA = np.log(y0)-B*np.log(x0)
    return lnA, B

def asSlopeIntercept(x0, x1, y0, y1):
    dy = y1 - y0
    dx = x1 - x0
    if dx == 0:
        m = 1.0
    else:
        m = dy / dx

    b = y0 - x0 * m
    return m, b

def evalDerivativeLinearGamma(m, b, g, x):
    return g * m * np.power(m*x+b, g-1.0)

class FullCurve:
    def __init__(self) -> None:
        self.Reset()
        
    def Reset(self):
        self.segments = [CurveSegment(), CurveSegment(), CurveSegment()]
        self.inv_segments = [CurveSegment(), CurveSegment(), CurveSegment()]
        self.W = 1.0
        self.invW = 1.0
        self.x0 = 0.25
        self.y0 = 0.25
        self.x1 = 0.75
        self.y1 = 0.75

    def Eval(self, x):
        x = x * self.invW
        return np.piecewise(x, [x < self.x0, (self.x0 < x) & (x < self.x1)], [
            lambda x: self.segments[0].Eval(x),
            lambda x: self.segments[1].Eval(x),
            lambda x: self.segments[2].Eval(x) 
        ])


class FilmicToneCurve:
    def __init__(self, user_params: CurveParamsUser) -> None:
        self.params = CurveParamsDirect()
        self.curve = FullCurve()

        # convert user params to curve params
        perceptual_gamma = 2.2

        toe_strength = np.clip(user_params.toe_strength, 0, 1)
        toe_length = np.clip(user_params.toe_length, 0, 1)**perceptual_gamma
        shoulder_strength = np.maximum(0, user_params.shoulder_strength) # np.clip(user_params.shoulder_strength, 0, 1)
        shoulder_length = np.clip(user_params.shoulder_length, 1e-5, 1) 

        shoulder_angle = np.clip(user_params.shoulder_angle, 0, 1)
        gamma = user_params.gamma

        x0 = toe_length * 0.5
        y0 = (1 - toe_strength) * x0

        remaining_y = 1 - y0

        initialW = x0 + remaining_y

        y1_offset = (1 - shoulder_length) * remaining_y
        x1 = x0 + y1_offset
        y1 = y0 + y1_offset

        extraW = np.exp2(shoulder_strength)-1

        W = initialW + extraW
        
        self.params.x0 = x0
        self.params.y0 = y0
        self.params.x1 = x1
        self.params.y1 = y1
        self.params.W = W
        self.params.gamma = gamma

        self.params.overshoot_x = (W * 2) * shoulder_angle * shoulder_strength
        self.params.overshoot_y = 0.5 * shoulder_angle * shoulder_strength


    def ToneMapping(self, x):
        self.curve.Reset()

        self.curve.W = self.params.W
        self.curve.invW = 1 / self.params.W

        self.params.x0 = self.params.x0 / self.params.W
        self.params.x1 = self.params.x1 / self.params.W
        self.params.overshoot_x = self.params.overshoot_x / self.params.W
        self.params.W = 1.0

        m, b = asSlopeIntercept(self.params.x0, self.params.x1, self.params.y0, self.params.y1)
        g = self.params.gamma

        self.curve.segments[1].offset_x = -(b/m)
        self.curve.segments[1].offset_y = 0.0
        self.curve.segments[1].scale_x = 1.0
        self.curve.segments[1].scale_y = 1.0
        self.curve.segments[1].lnA = g * np.log(m)
        self.curve.segments[1].B = g
        

        toeM = evalDerivativeLinearGamma(m,b,g,self.params.x0)
        shoulderM = evalDerivativeLinearGamma(m,b,g,self.params.x1)

        self.params.y0 = np.power(self.params.y0, self.params.gamma)
        self.params.y1 = np.power(self.params.y1, self.params.gamma)
        self.params.overshoot_y = np.power(1.0+self.params.overshoot_y, self.params.gamma) - 1.0

        self.curve.x0 = self.params.x0
        self.curve.x1 = self.params.x1
        self.curve.y0 = self.params.y0
        self.curve.y1 = self.params.y1


        self.curve.segments[0].offset_x=0
        self.curve.segments[0].offset_y=0
        self.curve.segments[0].scale_x=1
        self.curve.segments[0].scale_y=1
        self.curve.segments[0].lnA, self.curve.segments[0].B = solveAB(self.params.x0,self.params.y0,toeM)

        
        x0 = (1.0 + self.params.overshoot_x) - self.params.x1
        y0 = (1.0 + self.params.overshoot_y) - self.params.y1
        self.curve.segments[2].offset_x = 1.0 + self.params.overshoot_x
        self.curve.segments[2].offset_y = 1.0 + self.params.overshoot_y
        self.curve.segments[2].scale_x = -1
        self.curve.segments[2].scale_y = -1
        self.curve.segments[2].lnA, self.curve.segments[2].B = solveAB(x0, y0, shoulderM)

        # evaluate shoulder at the end of the curve
        scale = self.curve.segments[2].Eval(1.0)
        self.curve.segments[0].offset_y = self.curve.segments[0].offset_y / scale
        self.curve.segments[0].scale_y = self.curve.segments[0].scale_y / scale

        self.curve.segments[1].offset_y = self.curve.segments[1].offset_y / scale
        self.curve.segments[1].scale_y = self.curve.segments[1].scale_y / scale

        self.curve.segments[2].offset_y = self.curve.segments[2].offset_y / scale
        self.curve.segments[2].scale_y = self.curve.segments[2].scale_y / scale

        return self.curve.Eval(x)
