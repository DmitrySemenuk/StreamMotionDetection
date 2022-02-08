open class StreamMotionDetection {
    public init() {}
    
    public func backgroundSubstract(_ image: UIImage) -> UIImage {
        return OpenCVWrapper.backgroundSubstract(image)
    }
}
