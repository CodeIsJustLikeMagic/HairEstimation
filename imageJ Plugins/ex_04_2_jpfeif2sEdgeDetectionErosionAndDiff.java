import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_04_2_jpfeif2sEdgeDetectionErosionAndDiff implements PlugIn {
//edge detection with erosion and difference
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		int[][] structure = {{1,1,1},{1,1,1},{1,1,1}};
		ImagePlus res = erosion(imp,structure,8);
		//IJ.save(res, res.getTitle());
		res = differenze(imp,res,127);
		//IJ.save(res, res.getTitle());
		
	}

	
	private ImagePlus erosion(ImagePlus imp,int[][] structure, int md){
		//input is a binary grayscale image (0,255);
		ImageProcessor ip = imp.getProcessor();

		int w = ip.getWidth();
		int h = ip.getHeight();
		int size = structure.length;
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resImage = imp.duplicate();
		resImage.setTitle("ex_4_2_Erosion.png");
		ImageProcessor resIp = resImage.getProcessor();

		int[] c = new int[3]; 
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				c = ip.getPixel(u,v,c);
				if(c[0]==0) {//if the pixel is black we determine if it stays black or gets changed to white
					int z=0;
					for(int lineCounter = -(size/2); lineCounter <=(size/2); lineCounter++) {
						for(int rowCounter = -(size/2); rowCounter <=(size/2); rowCounter++) {
							c = ip.getPixel((u+rowCounter)%w,(v+lineCounter)%h,c);
							//newVal += gausWeight(rowCounter,lineCounter,o)*(double)c[0];
							if(structure[lineCounter+(size/2)][rowCounter+(size/2)]==1 &&c[0]==0) {
								//found a pixel in the structure element
								z++;
							}
						}
					}
					if(z<=md) { //make the pixel (u,v) white if we found a few black pixel in the structure element
						resIp.putPixel(u,v,255);
					}
					
				}// if the pixel is already white we dont do anything
				
			}
		}
		//display images	
		resImage.show();
		resImage.draw();
		return resImage;
		}
	
	private ImagePlus differenze(ImagePlus imp, ImagePlus imp2, int offset) {
		//imp-imp2 +offset
		//input are two Grayscale Images
		ImageProcessor ip1 = imp.getProcessor();
		ImageProcessor ip2 = imp2.getProcessor();
		
		int w = ip1.getWidth();
		int h = ip1.getHeight();
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resImage = IJ.createImage("ex_4_2_Differenze.png", "8-Bit",w,h,1);
		ImageProcessor resIp = resImage.getProcessor();

		int[] c1 = new int[3];
		int[] c2 = new int[3];
		int newVal=0;
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				c1 = ip1.getPixel(u,v,c1);
				c2 = ip2.getPixel(u,v,c2);
				newVal = Math.abs(c1[0]-c2[0])+offset;
				resIp.putPixel(u,v,newVal);
			}
		}
		//display images	
		resImage.show();
		resImage.draw();
		return resImage;
		}

}
