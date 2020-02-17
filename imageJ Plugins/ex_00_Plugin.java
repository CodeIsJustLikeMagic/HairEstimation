import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_00_Plugin implements PlugIn {

	public void run(String arg) {
		//save the current image in imp
		ImagePlus imp = IJ.getImage();	
		
		//use Image Converter to convert to Grayscale
		ImageConverter ic = new ImageConverter(imp);
		ic.convertToGray8();	

		//use Image Processor to find Edges
		ImageProcessor ip = imp.getProcessor();
		ip.findEdges();

		//display the new Image
		imp.updateAndDraw();
	}

}
