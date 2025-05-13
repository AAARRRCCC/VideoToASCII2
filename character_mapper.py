class CharacterMapper:
    def __init__(self):
        # Define a list of Japanese characters from dense to sparse
        # The order should be from characters that appear densest/brightest to least dense/darkest
        self.japanese_chars = [
            'あ', 'い', 'う', 'え', 'お', 
            'か', 'き', 'く', 'け', 'こ', 
            'さ', 'し', 'す', 'せ', 'そ', 
            'た', 'ち', 'つ', 'て', 'と', 
            'な', 'に', 'ぬ', 'ね', 'の', 
            'は', 'ひ', 'ふ', 'へ', 'ほ', 
            'ま', 'み', 'む', 'め', 'も', 
            'や', 'ゆ', 'よ', 
            'ら', 'り', 'る', 'れ', 'ろ', 
            'わ', 'を', 'ん', 
            'ア', 'イ', 'ウ', 'エ', 'オ', 
            'カ', 'キ', 'ク', 'ケ', 'コ', 
            'サ', 'シ', 'ス', 'セ', 'ソ', 
            'タ', 'チ', 'ツ', 'テ', 'ト', 
            'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 
            'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 
            'マ', 'ミ', 'ム', 'メ', 'モ', 
            'ヤ', 'ユ', 'ヨ', 
            'ラ', 'リ', 'ル', 'レ', 'ロ', 
            'ワ', 'ヲ', 'ン', 
            '゛', '゜', 'ー', '・', '「', '」', '（', '）'
        ]
        
        # Use a black dot for the darkest pixels
        self.black_char = '・'
        
        # Calculate brightness thresholds
        self.black_threshold = 30  # Pixels below this value will be represented by a black dot
        self.brightness_levels = len(self.japanese_chars))
    
        self.brightness_step = (255 - self.black_threshold) / self.brightness_levels
    
    def map_pixel_to_character(self, pixel_value):
        """
        Map a pixel value (0-255) to a Japanese character.
        
        Args:
            pixel_value (int): Grayscale pixel value (0-255)
            
        Returns:
            str: Japanese character representing the pixel brightness
        """
        # Black pixels get a dot
        if pixel_value <= self.black_threshold:
            return self.black_char
        
        # Map other pixel values to Japanese characters
        adjusted_value = pixel_value - self.black_threshold
        char_index = min(
            int(adjusted_value / self.brightness_step),
            len(self.japanese_chars) - 1
        )
        
        return self.japanese_chars[char_index]