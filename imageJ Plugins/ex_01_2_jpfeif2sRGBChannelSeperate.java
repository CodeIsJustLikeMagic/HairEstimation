import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_01_2_jpfeif2sRGBChannelSeperate implements PlugIn {
//split RGB image into 3 seperate color channels red green blue in seperate windows in each color 
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		split(imp.duplicate());
		
	}
	private void split(ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();
		
		int w = ip.getWidth();
		int h = ip.getHeight();
		int[] c = new int[3]; 
		//create Duplicate Images for red and green channel images
		ImagePlus blueImage = imp.duplicate();
		ImageProcessor blueIp = blueImage.getProcessor();
		ImagePlus greenImage = imp.duplicate();
		ImageProcessor greenIp = greenImage.getProcessor();
		
		//go through all the Pixels
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {	
				//save colors in c
				c = ip.getPixel(u,v,c);
				//construct color arrays for the channel images
				int[] red = {c[0],0,0};
				int[] green = {0,c[1],0};
				int[] blue = {0,0,c[2]};
				//assign new colors to the three channel images
				ip.putPixel(u,v,red);
				greenIp.putPixel(u,v,green);
				blueIp.putPixel(u,v,blue);
			}
		}
		//display the channel images
		imp.show();
		imp.draw();
		greenImage.show();
		greenImage.draw();
		blueImage.show();
		blueImage.draw();
	}
}
