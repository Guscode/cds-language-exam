# Assignment 3: Edge Detection

The goal of the assignment is to find text and characters using edge detection. The script detect_edges.py will take either one or multiple images and input, and output contoured images as well as text files generated with tesseract. 

### How to run

To run this code, please follow the guide for activating the virtual environment in [cds-visual-exam](https://github.com/Guscode/cds-visual-exam).

To test the script, in the virtual environment, please run:
```bash
cd Assignment_3
python detect_edges.py --image results/jefferson.jpg
```
This will return a contoured version of the original image as well as a text file generated with pytesseract.
For better results, you can include cropping coordinates to crop the image as closely to the text as possible, in order to reduce noise:
```bash
python detect_edges.py --image results/jefferson.jpg --crop-coordinates X750X700Y750Y1150
```

The results without cropping and the results with cropping:
<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_3/results/jeffersons.png" alt="Logo" width="1100" height="900">
</a>

The script also includes the possibility to perform edge-detection on multiple images.
```bash
python detect_edges.py --image-files signs
```
The results:

<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_3/results/city_signs.png" alt="Logo" width="900" height="900">
</a>



# User defined arguments

The user defined arguments are:

```bash
--image #Path to an image
--output #Path where you want the output files
--crop-coordinates # X and Y coordinates for cropping image in X1X2Y1Y2 format.
--image-files #Path to a folder with images. script takes all files from folder with .jpg, .jpeg or .png
--psm #Specifies page segmentation method. see below for specifications.

```

Page segmentation modes:

```bash
  --psm 0    Orientation and script detection (OSD) only.
  --psm 1    Automatic page segmentation with OSD.
  --psm 2    Automatic page segmentation, but no OSD, or OCR.
  --psm 3    Fully automatic page segmentation, but no OSD. (Default)
  --psm 4    Assume a single column of text of variable sizes.
  --psm 5    Assume a single uniform block of vertically aligned text.
  --psm 6    Assume a single uniform block of text.
  --psm 7    Treat the image as a single text line.
  --psm 8    Treat the image as a single word.
  --psm 9    Treat the image as a single word in a circle.
  --psm 10   Treat the image as a single character.
  --psm 11   Sparse text. Find as much text as possible in no particular order.
  --psm 12   Sparse text with OSD.
  --psm 13   Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
```

With the cropped jefferson image, it is fair to use psm 6, as it assumes a single block of vertically aligned text.

```bash
python detect_edges.py --image results/jefferson.jpg --crop-coordinates X750X700Y750Y1150 --psm 6 --output results
```

With the cropped image and psm 6, the output .txt file reads:

WE HOLD THESE TRUTHS TO BE SELF <br/>  
EVIDENT: THAT ALL MEN ARE CREATED <br/>
EQUAL, THAT. THEY ARB ENDOWED BY THEIR <br/>
CREATOR WITH CERTAIN INALIENABLE \ <br/>
RIGHTS, AMONG THESE ARE #IFE. LIBERTY <br/>
AND THE PURSUIT OF HAPPINESS. THAT <br/>
TO SECURE THESE RIGHTS _sittcabines a <br/>
ARE INSTIFUTED AMONG MEN. WE: <br/>
SOLEMNLY PUBLISH AND DECEARE THA] <br/>
THESE COLONIES ARE AND OF RIGHT


OUGHT TO BE FREE AND INDEPEN DENT <br/>
STATES: ‘AND FORTHE SUPPORT OF THIS <br/>
DECLARATION, WITH A FIRM RELIANCE <br/>
ON THE PROTECTION OF DIVINE <br/>
PROVIDENCE, WE MUTUALLY PLEDGE <br/>
OUR LIVES,/OUR FORTUNES-AND OUR <br/>

SACRED HONOUR. <br/>

