"""
Kornia Helper Module - Self-contained Kornia management

This module handles everything related to Kornia:
- Detection
- Installation
- Availability checking
- Version verification

All Kornia-related logic is centralized here for maintainability.
"""

import subprocess
import sys


class KorniaHelper:
    """
    Centralized Kornia management.
    
    Handles detection, installation, and version checking for Kornia.
    All Kornia-related logic is contained in this class.
    """
    
    _instance = None
    _kornia_available = None
    
    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Kornia helper"""
        if self._kornia_available is None:
            self._kornia_available = self._check_kornia()
    
    @staticmethod
    def _check_kornia():
        """Check if Kornia is available"""
        try:
            import kornia
            return True
        except ImportError:
            return False
    
    @property
    def is_available(self):
        """Check if Kornia is currently available"""
        if self._kornia_available is None:
            self._kornia_available = self._check_kornia()
        return self._kornia_available
    
    def install(self, auto_install=True, verbose=True):
        """
        Install Kornia if not available.
        
        Args:
            auto_install (bool): If True, automatically install. If False, only prompt.
            verbose (bool): Print installation progress
            
        Returns:
            bool: True if Kornia is now available, False otherwise
        """
        if self.is_available:
            return True
        
        if not auto_install:
            if verbose:
                print("\n" + "="*60)
                print("⚠️  Kornia not found - required for GPU augmentation")
                print("="*60)
                print("\nKornia provides 5-10x faster augmentation on GPU.")
                print("\nTo install manually:")
                print("  pip install kornia kornia-rs")
                print("\nOr enable auto-install: auto_install_kornia=True")
            return False
        
        if verbose:
            print("\n" + "="*60)
            print("📦 Installing Kornia for GPU augmentation...")
            print("="*60)
            print("\nThis will install: kornia kornia-rs")
            print("This is a one-time installation (~50MB).\n")
        
        try:
            # Install kornia and kornia-rs
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "kornia", "kornia-rs"],
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None
            )
            
            # Verify installation
            self._kornia_available = self._check_kornia()
            
            if self.is_available:
                if verbose:
                    print("✅ Kornia installed successfully!")
                    print("   GPU augmentation is now available.\n")
                return True
            else:
                if verbose:
                    print("❌ Installation completed but Kornia not detected.")
                    print("   Try manually: pip install kornia kornia-rs\n")
                return False
                
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"❌ Failed to install Kornia: {e}")
                print("   Try manually: pip install kornia kornia-rs\n")
            return False
        except Exception as e:
            if verbose:
                print(f"❌ Unexpected error: {e}")
                print("   Try manually: pip install kornia kornia-rs\n")
            return False
    
    def get_version(self):
        """Get Kornia version if available"""
        if not self.is_available:
            return None
        try:
            import kornia
            return kornia.__version__
        except Exception:
            return "unknown"
    
    def ensure_available(self, auto_install=True, verbose=True):
        """
        Ensure Kornia is available, installing if needed.
        
        Args:
            auto_install (bool): Whether to auto-install if missing
            verbose (bool): Print messages
            
        Returns:
            bool: True if Kornia is available, False otherwise
        """
        if self.is_available:
            return True
        
        return self.install(auto_install=auto_install, verbose=verbose)


# Global singleton instance
_kornia_helper = KorniaHelper()


def is_kornia_available():
    """
    Check if Kornia is available.
    
    Returns:
        bool: True if Kornia is available
    """
    return _kornia_helper.is_available


def install_kornia(auto_install=True, verbose=True):
    """
    Install Kornia if not available.
    
    Args:
        auto_install (bool): If True, automatically install. If False, only prompt.
        verbose (bool): Print installation progress
        
    Returns:
        bool: True if Kornia is now available
    """
    return _kornia_helper.install(auto_install=auto_install, verbose=verbose)


def ensure_kornia(auto_install=True, verbose=True):
    """
    Ensure Kornia is available, installing if needed.
    
    Args:
        auto_install (bool): Whether to auto-install if missing
        verbose (bool): Print messages
        
    Returns:
        bool: True if Kornia is available
    """
    return _kornia_helper.ensure_available(auto_install=auto_install, verbose=verbose)


def get_kornia_version():
    """
    Get Kornia version.
    
    Returns:
        str or None: Version string or None if not available
    """
    return _kornia_helper.get_version()

