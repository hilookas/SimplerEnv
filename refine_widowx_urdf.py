import os
import argparse
import xml.etree.ElementTree as ET
import trimesh

def scale_and_save_mesh(original_filepath, output_folder, scale_factor=1000.0):
    if not os.path.exists(original_filepath):
        raise FileNotFoundError(f"filepath not found: {original_filepath}")

    mesh = trimesh.load(original_filepath)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces) 
             for geom in mesh.geometry.values()]
        )
    mesh.vertices /= scale_factor

    os.makedirs(output_folder, exist_ok=True)

    basename = os.path.basename(original_filepath)
    new_filepath = os.path.join(output_folder, basename)
    
    mesh.export(new_filepath)
    return new_filepath

def process_urdf(input_urdf, output_urdf, mesh_output_folder, scale_factor=1000.0):

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    for mesh in root.iter('mesh'):
        filename_attr = mesh.get('filename')
        if filename_attr and (filename_attr.lower().endswith('.obj') or filename_attr.lower().endswith('.stl')):
            original_mesh_path = os.path.join(os.path.dirname(input_urdf), filename_attr)
            try:
                new_mesh_path = scale_and_save_mesh(original_mesh_path, mesh_output_folder, scale_factor)
            except Exception as e:
                continue

            new_rel_path = os.path.relpath(new_mesh_path, os.path.dirname(output_urdf))
            mesh.set('filename', new_rel_path)
            mesh.set('scale', "1 1 1")

    tree.write(output_urdf, encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="scale urdf and mesh files",
    )
    parser.add_argument("input_urdf", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/wx250s.urdf", help="输入的 URDF 文件路径")
    parser.add_argument("output_urdf", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/scale_wx250s.urdf", help="输出修改后的 URDF 文件路径")
    parser.add_argument("mesh_output_folder", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/scale_wx250s", help="缩放后的 mesh 文件保存的新文件夹")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="scle factor (default: 1000)")
    args = parser.parse_args()

    process_urdf(args.input_urdf, args.output_urdf, args.mesh_output_folder, args.scale_factor)
