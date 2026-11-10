import h5py
import os


DATASET_FOLDER = "datasets/nighttime_lights"


def show_item(name, obj):

    print(f"\n{name}")

    if isinstance(obj, h5py.Dataset):
        print("  TYPE: DATASET")
        print("  SHAPE:", obj.shape)
        print("  DATA TYPE:", obj.dtype)

    elif isinstance(obj, h5py.Group):
        print("  TYPE: GROUP")

    # Show attributes
    if hasattr(obj, "attrs"):

        for key, value in obj.attrs.items():

            print(
                f"  ATTRIBUTE: {key} = {value}"
            )


print("=" * 70)
print("NASA VIIRS NIGHTTIME LIGHT METADATA")
print("=" * 70)


files = [
    file_name
    for file_name in os.listdir(DATASET_FOLDER)
    if file_name.endswith(".h5")
]


for file_name in files:

    file_path = os.path.join(
        DATASET_FOLDER,
        file_name
    )

    print("\n")
    print("=" * 70)
    print("FILE:", file_name)
    print("=" * 70)

    try:

        with h5py.File(
            file_path,
            "r"
        ) as h5_file:

            h5_file.visititems(
                show_item
            )

    except Exception as error:

        print(
            "ERROR:",
            error
        )


print("\n")
print("=" * 70)
print("METADATA INSPECTION COMPLETE")
print("=" * 70)