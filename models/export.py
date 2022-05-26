import argparse
from utils.torch_utils import select_device

from models import Darknet
from utils.datasets import *
from utils.general import *


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov4.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    cfg, imgsz, weights = \
        opt.cfg, opt.img_size, opt.weights

    # Input
    device = select_device('cpu')
    # Load PyTorch model
    model = Darknet(cfg, imgsz)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    img = torch.zeros((opt.batch_size, model.channels, *opt.img_size), device=device)  # image size(1,1,320,192) iDetection
    
    # model = TempModel()
    # model = torch.load_state_dict(torch.load(opt.weights))
    model.eval()
    # model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)