import os
import av
import argparse
import pandas as pd
from tqdm import tqdm
from data.split_utils import is_blank_frame, try_read_qr_code, open_output_writer
from utils.files import find_files_recursively
from utils.logger import setup_basic_logger


log = setup_basic_logger(os.path.basename(__file__))


def split_fragments(
    fragments_filepaths: list[str],
    output_dir: str,
):
    current_item_id = None
    current_modifiers = None

    is_recording = False

    output_container = None
    output_stream = None
    current_start_pts = None

    buffer = []

    for fragment_filepath in tqdm(fragments_filepaths, desc="Processing fragments", unit="fragment"):

        # Open the video file
        with av.open(fragment_filepath) as input_container:
            input_stream = input_container.streams.video[0]

            # Decode input fragment frame by frame
            for frame_idx, input_frame in enumerate(input_container.decode(video=0)):

                # Convert frame to RGB
                frame_rgb = input_frame.to_ndarray(format="rgb24")

                # Skip over black frames
                if is_blank_frame(frame_rgb):
                    continue

                # Check if the frame is a separator frame
                metadata = try_read_qr_code(frame_rgb)

                if metadata is not None and isinstance(metadata, dict) and "item_id" in metadata:
                    # It is a metadata frame

                    # If we are currently recording, stop the recording
                    if is_recording:
                        # Flush and close current video
                        for packet in output_stream.encode():
                            packet.time_base = input_stream.time_base
                            output_container.mux(packet)
                        output_container.close()
                        is_recording = False

                    # Decode QR code
                    current_item_id = metadata["item_id"]
                    current_modifiers = metadata["modifiers"]

                    # Continue to next frame
                    continue

                else:
                    # This is not a metadata frame

                    # If we don't have any metadata yet, we will have to skip this frame
                    if current_item_id is None:
                        log.info(f"Need to skip frame {frame_idx} because we haven't seen a metadata frame yet.")
                        continue

                    # First frame of a new sequence
                    if not is_recording:
                        # Create new output writer
                        output_filename = f"{current_item_id}_{current_modifiers}.mp4"
                        output_filepath = os.path.join(output_dir, output_filename)
                        output_container, output_stream = open_output_writer(output_filepath, input_stream)
                        is_recording = True
                        current_start_pts = input_frame.pts
                        log.info(f"Starting new clip (item id: {current_item_id}, modifiers: {current_modifiers}).")

                        buffer.append({
                            "item_id": current_item_id,
                            "modifiers": current_modifiers,
                            "filename": output_filename,
                        })

                    # Create a new frame with the pixel values of the input frame
                    output_frame = av.VideoFrame.from_ndarray(input_frame.to_ndarray(), format=input_frame.format.name)

                    # Calculate new PTS relative to the first frame of the clip
                    rel_pts = input_frame.pts - current_start_pts
                    output_frame.pts = rel_pts

                    # Encode and write to output
                    try:
                        # Encode and mux
                        for packet in output_stream.encode(output_frame):
                            packet.time_base = input_stream.time_base
                            output_container.mux(packet)

                    except Exception as e:
                        log.exception(f"Error encoding frame {frame_idx} in {output_filepath}: {e}")
                        raise e

    # Close the last output file
    if is_recording:
        # Flush residue from this fragment
        for packet in output_stream.encode():
            packet.time_base = input_stream.time_base
            output_container.mux(packet)

        output_container.close()

    output_df = pd.DataFrame(buffer)
    return output_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate recorded frames")
    parser.add_argument("--input_dir", type=str, help="Directory where to search for fragments", default="/media/bene/getreal/video-streaming-linux/recordings/27005680-3421-436a-bdc1-06fecc993edd")
    parser.add_argument("--output_dir", type=str, help="Directory where to save the output fragments", default="/tmp")
    args = vars(parser.parse_args())

    fragments_filepaths = sorted(list(find_files_recursively(args["input_dir"], file_extensions=[".mkv"])))

    # fragments_attributes_df = extract_video_attributes(fragments_filepaths)
    # assert len(fragments_attributes_df["width"].unique()) == 1 and len(fragments_attributes_df["height"].unique()) == 1, "Expected all fragments to have the same width and height"

    output_videos_df = split_fragments(fragments_filepaths=fragments_filepaths, output_dir=args["output_dir"])
    log.info(f"Stored individual videos to \"{args['output_dir']}\"")

    output_csv_filepath = os.path.join(args["output_dir"], "video_clips.csv")
    output_videos_df.to_csv(output_csv_filepath, index=False)

