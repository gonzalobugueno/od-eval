import bpy
import itertools
import math
from mathutils import Vector
import os
import xml.etree.ElementTree as ET


# TUNABLE PARAMETERS
output_dir =  os.path.join(os.getcwd() ,'../datasets/synthetic-2')
annotations_path = os.path.join(output_dir, 'annotations.xml')


param_grid_synthetic = {
    'coin_x_metres': [-0.5, -0.25, 0, 0.25, 0.5],
    'coin_y_metres': [-0.1, 0, 0.25, 0.5],
    'coin_yaw': [math.radians(i) for i in range(0, 360, 45)],
    'sun_angle': [math.radians(0.53)],
    'sun_energy': [5, 10],
    'sun_rotation': [(15, 90), (75, 180), (15, 270)],
    'render_samples': [1],
    'res_factor': [1]
}

param_grid_simple = {
    'coin_x_metres': [-0.5, -0.25, 0, 0.25, 0.5],
    'coin_y_metres': [-0.1, 0, 0.25, 0.5],
    'coin_yaw': [0],
    'sun_angle': [math.radians(0.53)],
    'sun_energy': [10],
    'sun_rotation': [(75, 180)],
    'render_samples': [1],
    'res_factor': [1]
}

pg = param_grid_simple

# END TUNABLE PARAMETERS

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def grid_search(params):
    keys = list(params.keys())
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))

annotations = ET.Element('annotations')
version = ET.SubElement(annotations, 'version')
version.text = '1.1'


def make_image_element(idx, width, height):
    image_elem = ET.Element('image', {
        'id': str(idx),
        'name': f'{idx}.png',
        'width': str(width),
        'height': str(height)
    })
    return image_elem


def bbox_to_box_element(coords, label='50cent'):
    x1, y1, x2, y2 = coords
    box_attribs = {
        'label': label,
        'occluded': '0',
        'xtl': str(x1),
        'ytl': str(y1),
        'xbr': str(x2),
        'ybr': str(y2)
    }
    return ET.Element('box', box_attribs)


def get_object_render_bbox(obj):
    render = bpy.context.scene.render
    scale = render.resolution_percentage / 100
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)
    camera = bpy.context.scene.camera
    view_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=width, y=height
    )
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        camera_corner = view_matrix @ world_corner
        clip_corner = projection_matrix @ camera_corner.to_4d()
        if clip_corner.w <= 0:
            continue
        ndc = clip_corner / clip_corner.w
        x = (ndc.x * 0.5 + 0.5) * width
        y = (1 - (ndc.y * 0.5 + 0.5)) * height
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)
    if min_x == float('inf'):
        return None
    return (min_x, min_y, max_x, max_y)


ensure_dir(output_dir)
ensure_dir(os.path.join(output_dir, 'images'))

bpy.ops.wm.open_mainfile(filepath="my.blend")

scene = bpy.context.scene
obj = bpy.data.objects['50cent']
sun = scene.objects['Sun']
CAM = bpy.data.objects['Camera']
scene.camera = CAM

for idx, grid in enumerate(grid_search(pg)):
    obj.location.x = grid['coin_x_metres']
    obj.location.y = grid['coin_y_metres']
    obj.rotation_euler[2] = grid['coin_yaw']
    elev, azim = grid['sun_rotation']
    sun.rotation_euler = (elev, 0, azim)
    sun.data.angle = grid['sun_angle']
    sun.data.energy = grid['sun_energy']

    w = 1920 * grid['res_factor']
    h = 1080 * grid['res_factor']
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.image_settings.file_format = 'PNG'
    out_img = os.path.join(output_dir, 'images', f'{idx}.png')
    scene.render.filepath = out_img
    scene.render.film_transparent = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.eevee.taa_render_samples = 1

    bpy.ops.render.render(write_still=True)

    bbox = get_object_render_bbox(obj)
    if bbox:
        img_elem = make_image_element(idx, w, h)
        annotations.append(img_elem)
        box_elem = bbox_to_box_element(bbox)
        img_elem.append(box_elem)

tree = ET.ElementTree(annotations)
tree.write(annotations_path, encoding='utf-8', xml_declaration=True)
print(f"Saved CVAT 1.1 annotations to {annotations_path}")
