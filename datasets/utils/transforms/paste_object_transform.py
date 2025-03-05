import torch
from PIL import ImageDraw

class ModifierPaste:
    def __init__(self, shape='square', size=18, color='magenta', position='top_left', probability=0.7):
        self.shape = shape
        self.size = size
        self.color = color
        self.position = position
        self.probability = probability

        if self.position == 'top_left':
            base_x = 20
            base_y = 20
        elif self.position == 'random':
            base_x = torch.randint(0, 224 - self.size, (1,)).item()
            base_y = torch.randint(0, 224 - self.size, (1,)).item()
        else:
            raise NotImplementedError

        if self.shape == 'square':
            self.img_draw_params = {'xy': (
                (base_x, base_y), (base_x + self.size, base_y), (base_x + self.size, base_y + self.size),
                (base_x, base_y + self.size)),
                'fill': self.color,
                'width': 2}
        else:
            raise NotImplementedError

    def update_params(self):
        if self.color == 'random':
            self.img_draw_params['fill'] = (
            torch.randint(0, 255, (1,)).item(), torch.randint(0, 255, (1,)).item(), torch.randint(0, 255, (1,)).item())
        if self.position == 'random':
            base_x = torch.randint(0, 224 - self.size, (1,)).item()
            base_y = torch.randint(0, 224 - self.size, (1,)).item()
            self.img_draw_params['xy'] = (
                (base_x, base_y), (base_x + self.size, base_y), (base_x + self.size, base_y + self.size),
                (base_x, base_y + self.size))

    def __call__(self, img):
        tmp = torch.rand(1)
        if self.probability > 0 and tmp < self.probability:
            self.update_params()
            draw = ImageDraw.Draw(img)
            if self.shape == 'square':
                draw.polygon(**self.img_draw_params)

        return img