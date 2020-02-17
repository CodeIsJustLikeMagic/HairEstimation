import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_02_1_jpfeif2sGrayscaleRescale implements PlugIn {
//rgb to grayscale. scale down grayvalues to [100,200]
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		grayscale_100_200(imp.duplicate());
		
		
	}
	private void grayscale_100_200 (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();

		
		int w = ip.getWidth();
		int h = ip.getHeight();
		
		//create blank 8-bit grayscale images in the correct size
		ImagePlus grayImage = IJ.createImage("ex_02_1_Nachher (Grauwertstauchung)", "8-bit grayscale",w,h,1);
		ImageProcessor gIp = grayImage.getProcessor();

		int[] c = new int[3]; 
		//go through all the pixels
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				//get colors
				c = ip.getPixel(u,v,c);

				//grayscale with equal parts from all colors
				//adjusting values to b_low = 0, b_high=255
				int g = (int) ( ((double) (c[0]+c[1]+c[2])) / 3);
				g = (int) ((g *(100.0/255.0)) +100);
				gIp.putPixel(u,v,g);
				
			}
		}
		//display images	
		grayImage.show();
		grayImage.draw();
		}
}
