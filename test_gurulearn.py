"""
Test script to verify gurulearn library functionality.

Run with: python test_gurulearn.py
"""

import sys
import time
from pathlib import Path

# Add parent to path for testing
sys.path.insert(0, str(Path(__file__).parent))


def test_lazy_loading():
    """Test that imports are lazy and don't load heavy dependencies."""
    print("\n" + "=" * 50)
    print("TEST 1: Lazy Loading")
    print("=" * 50)
    
    # Clear any cached imports
    modules_before = set(sys.modules.keys())
    
    start = time.time()
    import gurulearn
    import_time = time.time() - start
    
    print(f"✓ Import time: {import_time:.3f}s")
    
    # Check that heavy modules are NOT loaded
    heavy_modules = ["torch", "tensorflow", "langchain", "librosa"]
    loaded_heavy = [m for m in heavy_modules if m in sys.modules]
    
    if loaded_heavy:
        print(f"✗ Heavy modules loaded during import: {loaded_heavy}")
    else:
        print("✓ Heavy modules NOT loaded during import (lazy loading works)")
    
    # Check version
    print(f"✓ Version: {gurulearn.__version__}")
    
    return import_time < 1.0 and len(loaded_heavy) == 0


def test_flowbot():
    """Test FlowBot module with sample data."""
    print("\n" + "=" * 50)
    print("TEST 2: FlowBot Module")
    print("=" * 50)
    
    try:
        import pandas as pd
        from gurulearn import FlowBot
        
        # Create sample data
        df = pd.DataFrame({
            'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing'],
            'brand': ['Apple', 'Samsung', 'Nike', 'Adidas'],
            'product': ['iPhone', 'Galaxy', 'Shoes', 'Sneakers'],
            'price': [999, 899, 120, 110]
        })
        
        # Initialize bot
        bot = FlowBot(df)
        bot.add('category', 'Select a category:')
        bot.add('brand', 'Select a brand:')
        bot.finish('product', 'price')
        
        # Test validation
        errors = bot.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"
        print("✓ FlowBot validation passed")
        
        # Test process flow
        response = bot.process('user1', '')
        assert 'message' in response
        assert 'suggestions' in response
        print(f"✓ Initial response: {response['message'][:50]}...")
        
        # Select category
        response = bot.process('user1', 'Electronics')
        assert 'suggestions' in response
        print(f"✓ After category selection: {response['suggestions']}")
        
        # Select brand
        response = bot.process('user1', 'Apple')
        assert 'completed' in response or 'results' in response
        print("✓ Flow completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ FlowBot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ct_scan_processor():
    """Test CTScanProcessor module."""
    print("\n" + "=" * 50)
    print("TEST 3: CTScanProcessor Module")
    print("=" * 50)
    
    try:
        from gurulearn import CTScanProcessor
        import numpy as np
        
        # Initialize processor
        processor = CTScanProcessor(kernel_size=5, clip_limit=2.0)
        print("✓ CTScanProcessor initialized")
        
        # Test image processing methods with synthetic data
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Test sharpen
        sharpened = processor.sharpen(test_image)
        assert sharpened.shape == test_image.shape
        print("✓ Sharpen method works")
        
        # Test median denoise
        denoised = processor.median_denoise(test_image)
        assert denoised.shape == test_image.shape
        print("✓ Median denoise method works")
        
        # Test contrast enhancement
        enhanced = processor.enhance_contrast(test_image)
        assert enhanced.shape == test_image.shape
        print("✓ Contrast enhancement works")
        
        # Test quality evaluation
        metrics = processor.evaluate_quality(test_image, enhanced)
        assert metrics.psnr > 0
        assert metrics.snr is not None
        print(f"✓ Quality metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ CTScanProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_classifier():
    """Test ImageClassifier module initialization."""
    print("\n" + "=" * 50)
    print("TEST 4: ImageClassifier Module")
    print("=" * 50)
    
    try:
        from gurulearn import ImageClassifier
        
        # Initialize classifier
        classifier = ImageClassifier()
        print(f"✓ ImageClassifier initialized on device: {classifier.device}")
        
        # Test that we can get optimal workers
        workers = classifier._get_optimal_workers()
        print(f"✓ Optimal workers: {workers}")
        
        # Test model building (simple CNN doesn't require pretrained weights)
        model = classifier._build_simple_cnn(num_classes=10)
        assert model is not None
        print("✓ Simple CNN model built successfully")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {num_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ ImageClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_recognition():
    """Test AudioRecognition module initialization."""
    print("\n" + "=" * 50)
    print("TEST 5: AudioRecognition Module")
    print("=" * 50)
    
    try:
        from gurulearn import AudioRecognition
        import numpy as np
        
        # Initialize
        audio = AudioRecognition(sample_rate=16000, n_mfcc=20)
        print("✓ AudioRecognition initialized")
        
        # Test feature extraction with synthetic audio
        synthetic_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
        features = audio.extract_features(synthetic_audio)
        
        assert features.ndim == 2
        print(f"✓ Feature extraction works: shape={features.shape}")
        
        # Test augmentation
        augmented = audio.augment_audio(synthetic_audio, sr=16000)
        assert len(augmented) == 3
        print(f"✓ Audio augmentation works: {len(augmented)} variants created")
        
        return True
        
    except Exception as e:
        print(f"✗ AudioRecognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_model_analysis():
    """Test MLModelAnalysis module with sample data."""
    print("\n" + "=" * 50)
    print("TEST 6: MLModelAnalysis Module")
    print("=" * 50)
    
    try:
        from gurulearn import MLModelAnalysis
        import pandas as pd
        import numpy as np
        from pathlib import Path
        import tempfile
        
        # Initialize
        analyzer = MLModelAnalysis(task_type="auto", auto_feature_engineering=True)
        print("✓ MLModelAnalysis initialized")
        
        # Create sample regression data
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randn(n_samples) * 10 + 50
        })
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test training
            result = analyzer.train_and_evaluate(
                csv_file=temp_path,
                target_column='target',
                test_size=0.3,
                model_name='linear_regression'
            )
            
            assert result.metrics.test_r2 is not None
            print(f"✓ Model trained: R² = {result.metrics.test_r2:.4f}")
            
            # Test prediction
            pred = analyzer.predict({'feature1': 0.5, 'feature2': -0.3, 'category': 'A'})
            print(f"✓ Prediction: {pred[0]:.2f}")
            
        finally:
            Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ MLModelAnalysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependency_check():
    """Test the dependency checking functionality."""
    print("\n" + "=" * 50)
    print("TEST 7: Dependency Checking")
    print("=" * 50)
    
    try:
        import gurulearn
        
        status = gurulearn.check_dependencies(verbose=False)
        
        available = sum(1 for v in status.values() if v)
        total = len(status)
        
        print(f"✓ Dependencies checked: {available}/{total} available")
        
        # Show missing optional dependencies
        missing = [k for k, v in status.items() if not v]
        if missing:
            print(f"  Optional missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency check failed: {e}")
        return False


def test_ocr_module():
    """Test OCR module imports and model instantiation."""
    print("\n" + "=" * 50)
    print("TEST 8: OCR Module")
    print("=" * 50)
    
    try:
        # Test imports
        from gurulearn.ocr import (
            split_datasets,
            merge_datasets,
            rebalance_splits,
            shuffle_augment,
            load_class_names,
            OCRTrainer,
            OCRPredictor,
            OCRPipeline,
            VGG_OCR,
            SplitResult,
            MergeResult,
            RebalanceResult,
            AugmentResult,
            TrainHistory,
            EvalResult,
            PredictionResult,
            PipelineResult,
        )
        print("✓ All OCR imports successful")
        
        # Test model instantiation with various class counts
        for nc in [10, 26, 36]:
            model = VGG_OCR(num_classes=nc + 1)  # +1 for CTC blank
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ VGG_OCR({nc} classes): {params:,} parameters")
        
        # Test load_class_names with a temp data.yaml
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                          delete=False, dir='.') as f:
            f.write("train: train/images\n")
            f.write("val: valid/images\n")
            f.write("nc: 5\n")
            f.write("names: ['A', 'B', 'C', 'D', 'E']\n")
            tmp_yaml = f.name
        
        try:
            names = load_class_names(tmp_yaml)
            assert names == ['A', 'B', 'C', 'D', 'E']
            print(f"✓ load_class_names: {names}")
        finally:
            Path(tmp_yaml).unlink()
        
        # Test dataclass creation
        sr = SplitResult()
        mr = MergeResult()
        rr = RebalanceResult()
        ar = AugmentResult()
        th = TrainHistory()
        er = EvalResult()
        pr = PredictionResult(text="ABC")
        plr = PipelineResult()
        print("✓ All dataclasses instantiate correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#" * 50)
    print("# GURULEARN v5.1 TEST SUITE")
    print("#" * 50)
    
    tests = [
        ("Lazy Loading", test_lazy_loading),
        ("FlowBot", test_flowbot),
        ("CTScanProcessor", test_ct_scan_processor),
        ("ImageClassifier", test_image_classifier),
        ("AudioRecognition", test_audio_recognition),
        ("MLModelAnalysis", test_ml_model_analysis),
        ("Dependency Check", test_dependency_check),
        ("OCR Module", test_ocr_module),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
