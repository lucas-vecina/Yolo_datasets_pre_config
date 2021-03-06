from pathlib import Path
import json
from collections import defaultdict
import os
import shutil
from multiprocessing import Pool
from functools import partial
import random
import tqdm
import copy

def clean_export_dir(darknet_export_images_dir: Path, darknet_export_labels_dir: Path, output_export_base: Path):
    shutil.rmtree(output_export_base, ignore_errors=True)
    darknet_export_images_dir.mkdir(parents=True)
    darknet_export_labels_dir.mkdir(parents=True)

def copy_image(darknet_export_images_dir: Path, src_file: Path, new_file_name: str):
    old_file_name = src_file.name
    shutil.copy(src_file, darknet_export_images_dir)

    dst_file = darknet_export_images_dir / old_file_name
    new_dst_file_name = darknet_export_images_dir / new_file_name
    os.rename(dst_file, new_dst_file_name)

def convert_object_entry(
    obj: dict,
    image_width: float,
    image_height: float,
    class_id_mapping: dict
):
    class_title = obj["classTitle"]
    class_id = class_id_mapping[class_title]

    left, top = obj["points"]["exterior"][0]
    right, bottom = obj["points"]["exterior"][1]

    mid_x = (left + right) / 2
    mid_y = (top + bottom) / 2

    bb_width = right - left
    bb_height = bottom - top

    norm_x = mid_x / image_width
    norm_y = mid_y / image_height

    norm_bb_width = bb_width / image_width
    norm_bb_height = bb_height / image_height

    if not (
        (0 <= norm_x <= 1)
        or (0 <= norm_y <= 1)
        or (0 <= norm_bb_width <= 1)
        or (0 <= norm_bb_height <= 1)
    ):
        raise RuntimeWarning(
            f"Normalized bounding box values outside the valid range! "
            f"x = {norm_x}; y = {norm_y}; w = {norm_bb_width}; h = {norm_bb_height}"
        )

    return class_id, class_title, norm_x, norm_y, norm_bb_width, norm_bb_height


def write_meta_data(
    output_export_base: Path,
    class_id_mapping: dict,
    num_labeled_images: int,
    class_counter: dict,
):
    # write class id mapping
    with open(output_export_base / "classes.txt", "w") as class_info_file:

        for class_name, _ in sorted(class_id_mapping.items(), key=lambda kv: kv[1]):
            class_info_file.write("{}\n".format(class_name))

    # write stats
    print("Number of exported Images: {} ".format(num_labeled_images))

    for class_name, count in sorted(
        class_counter.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"{class_name} -> {count}")

    with open(output_export_base / "stats.txt", "w") as class_stat_file:

        class_stat_file.write("Number of images: {}\n\n".format(num_labeled_images))
        class_stat_file.write("Objects per class:\n")

        total_num_objects = 0

        for class_name, count in sorted(
            class_counter.items(), key=lambda kv: kv[1], reverse=True
        ):
            total_num_objects += count
            class_stat_file.write("{} -> {}\n".format(class_name, count))

        class_stat_file.write(
            "\nTotal number of objects: {}\n".format(total_num_objects)
        )


def convert_label(
    darknet_export_images_dir: Path,
    darknet_export_labels_dir: Path,
    class_id_mapping: dict,
    label: Path
):
    class_counter = defaultdict(int)
    name = label.stem
    image = Path(str(label).replace("/ann/", "/img/").replace(".json", ""))

    with open(label) as json_file:
        data = json.load(json_file)

        if len(data["objects"]) > 0:

            image_width = data["size"]["width"]
            image_height = data["size"]["height"]

            copy_image(darknet_export_images_dir, image, name)
            label_file_name = darknet_export_labels_dir / f"{Path(name).stem}.txt"
            exc_obj_name = darknet_export_images_dir.parent / "excluded_obj.txt"

            with open(label_file_name, "w") as darknet_label:

                for obj in data["objects"]:
                    if(obj["classTitle"]) in class_id_mapping:
                        try:
                            (
                                class_id,
                                class_title,
                                norm_x,
                                norm_y,
                                norm_bb_width,
                                norm_bb_height,
                            ) = convert_object_entry(
                                obj,
                                image_height=image_height,
                                image_width=image_width,
                                class_id_mapping=class_id_mapping
                                )

                            class_counter[class_title] += 1

                            darknet_label.write(
                                "{} {} {} {} {}\n".format(
                                    class_id,
                                    norm_x,
                                    norm_y,
                                    norm_bb_width,
                                    norm_bb_height,
                                )
                            )
                        except :
                            print(f"[Warning] Failed to convert object entry in {label_file_name} \n")
                    else:
                        with open(exc_obj_name, 'a') as exc:
                            tg = obj["classTitle"]
                            exc.write(f"Object not included =={tg}== detected on {name} \n")
                        

    return class_counter

def get_paths():
    with open("paths.json", 'r') as file:
        data = json.load(file)
        sly = data["DATASET_PATH"]
        dkn = data["DARKNET_PATH"]
        file.close()

    out = Path.cwd() / "output"

    return Path(sly), Path(out), Path(dkn)

def get_classes():
    class_id_mapping = {}
    i = 0

    with open("classes.json", 'r') as classes:
        active_classes = json.load(classes)

        for d in active_classes["classes"]:
            if(d["active"] == 1):
                class_id_mapping[d["title"]] = i
                i += 1
    
    return class_id_mapping

def move_output(imgs_list: list, darknet_base: Path, file: Path, output_export_base: Path):
    destination = output_export_base / file.stem
    destination.mkdir(parents=True)

    with open(output_export_base / file, 'w') as f:
        for img in imgs_list:
            try:
                shutil.move(str(output_export_base / "images" / img.stem), destination)
                shutil.move(str(output_export_base / "labels" / Path(Path(img.stem).stem + ".txt")), destination)
                f.write(str(darknet_base / "data" / file.stem / img.stem) + "\n")
            except FileNotFoundError:
                pass

def generate_train_test(output_export_base: Path, labels: list, darknet_ref: Path):
    with open("classes.json", 'r') as jsonfile:
        data = json.load(jsonfile)
        train_p = data["specs"]["train"]
        val_p = data["specs"]["validation"]
        jsonfile.close()

    num_imgs = len(labels)
    train_imgs = random.sample(labels, int(num_imgs * train_p))

    labels_gp1 = list(set(labels) - set(train_imgs))
    val_imgs = random.sample(labels_gp1, int(num_imgs * val_p))

    test_imgs = list(set(labels_gp1) - set(val_imgs))

    move_output(train_imgs, darknet_ref, Path("train.txt"), output_export_base)
    move_output(val_imgs, darknet_ref, Path("val.txt"), output_export_base)
    move_output(test_imgs, darknet_ref, Path("test.txt"), output_export_base)

    shutil.rmtree(output_export_base / "images", ignore_errors=True)
    shutil.rmtree(output_export_base / "labels", ignore_errors=True)

def main():
    sly_base, output_export_base, darknet_ref = get_paths()
    class_id_mapping = get_classes()

    darknet_export_images_dir = output_export_base / "images"
    darknet_export_labels_dir = output_export_base / "labels"
    labels = list(sly_base.glob("*/ann/*.json"))

    clean_export_dir(darknet_export_images_dir, darknet_export_labels_dir, output_export_base)

    convert_func = partial(
        convert_label,
        darknet_export_images_dir,
        darknet_export_labels_dir,
        class_id_mapping,
    )

    global_class_counter = defaultdict(int)

    with Pool() as p:
        for class_counter in tqdm.tqdm(p.imap_unordered(convert_func, labels)):
            for class_name, count in class_counter.items():
                global_class_counter[class_name] += count

    write_meta_data(output_export_base, class_id_mapping, len(labels), global_class_counter)
    generate_train_test(output_export_base, labels, darknet_ref)

if __name__ == "__main__":
    main()
